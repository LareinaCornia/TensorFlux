import asyncio

import torch

from efficientserving import utils
from efficientserving.generation import (
    generate,
    generate_kv,
    generate_padded,
    generate_padded_kv,
)
from efficientserving.model import GPT
from efficientserving.server.interface import InferenceRequest, InferenceServer


class StaticBatchServer(InferenceServer):
    """Instantiation of InferenceServer with static batching technique.

    Students must fill in custom_scheduler_step and custom_cleanup
    according to the static batching spec in the handout.
    """

    def __init__(self, model: GPT, device: str, use_kv: bool = False,
                 batch_size: int = 1, max_wait_time_s: float = 60.0):
        super().__init__(model, device, use_kv)
        self.batch_size = batch_size
        self.max_wait_time_s = max_wait_time_s
        self.ctx = utils.get_torch_ctx(device)

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_wait_time_s < 0:
            raise ValueError("max_wait_time_s must be non-negative")

    async def _next_request(self) -> InferenceRequest | None:
        get_task = asyncio.create_task(self.queue.get())
        stop_task = asyncio.create_task(self._stop_signal.wait())
        done, pending = await asyncio.wait(
            (get_task, stop_task), return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        return get_task.result() if get_task in done else None

    async def _collect_batch(self, first: InferenceRequest) -> list[InferenceRequest]:
        batch = [first]
        deadline = first.arrival_time + self.max_wait_time_s
        loop = asyncio.get_running_loop()
        while len(batch) < self.batch_size:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            get_task = asyncio.create_task(self.queue.get())
            stop_task = asyncio.create_task(self._stop_signal.wait())
            done, pending = await asyncio.wait(
                (get_task, stop_task), timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if get_task in done:
                batch.append(get_task.result())
            else:
                break
        return batch

    def _run_batch(self, batch: list[InferenceRequest]) -> list[torch.Tensor]:
        prompt_width = max(req.idx.size(1) for req in batch)
        max_new_tokens = max(req.max_new_tokens for req in batch)

        sampling = {(req.temperature, req.top_k) for req in batch}
        if len(sampling) != 1:
            results: list[torch.Tensor | None] = [None] * len(batch)
            for parameters in sampling:
                indices = [
                    i for i, req in enumerate(batch)
                    if (req.temperature, req.top_k) == parameters
                ]
                group_results = self._run_batch([batch[i] for i in indices])
                for i, result in zip(indices, group_results):
                    results[i] = result
            return [result for result in results if result is not None]
        temperature, top_k = next(iter(sampling))

        padded = torch.zeros(
            (len(batch), prompt_width), dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros_like(padded, dtype=torch.bool)
        for row, req in enumerate(batch):
            prompt = req.idx[0].to(self.device)
            padded[row, -prompt.numel():] = prompt
            attention_mask[row, -prompt.numel():] = True

        with torch.no_grad(), self.ctx:
            if all(req.idx.size(1) == prompt_width for req in batch):
                generate_fn = generate_kv if self.use_kv else generate
                output = generate_fn(
                    self.model, padded, max_new_tokens,
                    temperature=temperature, top_k=top_k,
                )
            else:
                generate_fn = generate_padded_kv if self.use_kv else generate_padded
                output = generate_fn(
                    self.model, padded, attention_mask, max_new_tokens,
                    temperature=temperature, top_k=top_k,
                )
        utils.synchronize(self.device)

        results = []
        for row, req in enumerate(batch):
            generated = output[row, prompt_width:prompt_width + req.max_new_tokens]
            generated = generated.to(req.idx.device)
            results.append(torch.cat((req.idx[0], generated)).unsqueeze(0))
        return results

    def _finish_batch(self, batch: list[InferenceRequest]):
        try:
            results = self._run_batch(batch)
        except BaseException as exc:
            for req in batch:
                self._fail(req, exc)
            return
        for req, result in zip(batch, results):
            self._complete(req, result)

    async def custom_scheduler_step(self):
        first = await self._next_request()
        if first is None:
            return
        batch = await self._collect_batch(first)
        self._finish_batch(batch)

    async def custom_cleanup(self):
        while not self.queue.empty():
            batch = []
            while len(batch) < self.batch_size and not self.queue.empty():
                batch.append(self.queue.get_nowait())
            if batch:
                self._finish_batch(batch)
