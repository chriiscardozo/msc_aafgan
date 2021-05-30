import os
from Utils import cuda_utils, commons_utils
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid_master.pytorch_fid.inception import InceptionV3
import pytorch_fid_master.pytorch_fid.fid_score as pytorch_fid_scores

class FID:
    def __init__(self, real_samples, double_tensors=False, device=cuda_utils.DEVICE, batch_size=128, dims=2048):
        self.real_samples = real_samples
        self.device = device
        self.batch_size = batch_size
        self.dims = dims

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)

        if double_tensors: self.model = self.model.double()

        result = self._evaluate_in_model(real_samples)
        self.mu_real = np.mean(result, axis=0)
        self.sigma_real = np.cov(result, rowvar=False)


    def _evaluate_in_model(self, samples):
        # if batch_size > len(dataset) or len(dataset) % batch_size != 0:
        #     raise Exception("batch_size must be smaller than dataset size and dataset size should be divisible by batch_size: bs=" + str(batch_size) + ";ds=" + str(len(dataset)))
        self.model.eval()
        dl = torch.utils.data.DataLoader(samples, batch_size=self.batch_size, drop_last=False, num_workers=commons_utils.cpu_count())
        pred_arr = np.empty((len(samples), self.dims))
        start_idx = 0

        for batch in tqdm(dl):
            batch = batch.to(self.device)
            with torch.no_grad():
                pred = self.model(batch)[0]
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]
        return pred_arr


    def _calculate_fid(self, fake_samples):
        result = self._evaluate_in_model(fake_samples)
        mu_fake = np.mean(result, axis=0)
        sigma_fake = np.cov(result, rowvar=False)
        return pytorch_fid_scores.calculate_frechet_distance(self.mu_real, self.sigma_real, mu_fake, sigma_fake)


    def _load_fake_samples_from_file(self, path_to_fake_samples):
        fake_samples = []
        for filename in os.listdir(path_to_fake_samples):
            if 'npz' not in filename: continue
            fake_samples.append(np.load(os.path.join(path_to_fake_samples, filename))["generatedImages"])
        fake_samples = np.concatenate(fake_samples)
        return fake_samples


    def calculate_fid(self, output_seed_dir, epoch):
        path_to_fake_samples = os.path.join(output_seed_dir, commons_utils.DIR_SAMPLES_CSV, str(epoch))
        fake_samples = self._load_fake_samples_from_file(path_to_fake_samples)

        return self._calculate_fid(fake_samples)
