import time
from abc import abstractmethod, ABC

import torch
import ignite.metrics
import ignite.engine

doTimer = True

# device = torch.device("cuda" if torch.cuda.is_available()
#                           else "mps" if torch.backends.mps.is_available()
#                           else "cpu")
device = "cpu"


class Metric(ABC):
    @staticmethod
    @abstractmethod
    def evaluate(image1: torch.tensor, image2: torch.tensor) -> float:
        pass

    @staticmethod
    @abstractmethod
    def metric_name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def higher_is_better() -> bool:
        pass


class PSNR(Metric):
    @staticmethod
    def evaluate(image1: torch.tensor, image2: torch.tensor) -> float:
        start_time = time.process_time()

        def eval_step(engine, batch):
            return batch

        evaluator = ignite.engine.Engine(eval_step)

        psnr = ignite.metrics.PSNR(data_range=255., device=device)

        psnr.attach(evaluator, "psnr")
        state = evaluator.run([[image2.type(torch.float32), image1.type(torch.float32)]])
        result = state.metrics["psnr"]
        end_time = time.process_time()
        if doTimer:
            print(f"The PSNR calculation took {end_time - start_time} seconds.")
        return result

    @staticmethod
    def metric_name() -> str:
        return "PSNR"

    @staticmethod
    def higher_is_better() -> bool:
        return True


class SSIM(Metric):

    @staticmethod
    def evaluate(image1: torch.tensor, image2: torch.tensor) -> float:
        start_time = time.process_time()
        
        def eval_step(engine, batch):
            return batch

        evaluator = ignite.engine.Engine(eval_step)

        ssim = ignite.metrics.SSIM(data_range=1.0, device=device)

        ssim.attach(evaluator, "ssim")

        print(image2)

        i2 = image2.transpose(0,2).transpose(1,2).unsqueeze(0)
        i1 = image1.transpose(0,2).transpose(1,2).unsqueeze(0)

        print(i2.size())


        state = evaluator.run([[i2, i1]])
        result = state.metrics["ssim"]




        end_time = time.process_time()
        if doTimer:
            print(f"The SSIM calculation took {end_time - start_time} seconds.")
        return result

    @staticmethod
    def metric_name() -> str:
        return "SSIM"

    @staticmethod
    def higher_is_better() -> bool:
        return True
