import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import train_harder_env
from models.encoder import CNNEncoder
from models.transition import TransitionModel
from training import train_world_model


class DeviceSelectionTests(unittest.TestCase):
    def test_select_device_prefers_mps_when_available(self):
        with mock.patch.object(train_world_model.torch.cuda, "is_available", return_value=False):
            with mock.patch.object(
                train_world_model.torch.backends.mps, "is_available", return_value=True
            ):
                device = train_world_model.select_device()

        self.assertEqual(device.type, "mps")


class ResumeCheckpointTests(unittest.TestCase):
    def test_load_training_state_restores_epoch_and_best_loss(self):
        encoder = CNNEncoder(latent_dim=8)
        transition = TransitionModel(latent_dim=8)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(transition.parameters()), lr=1e-3
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "epoch": 7,
                    "loss": 0.125,
                    "encoder_state_dict": encoder.state_dict(),
                    "transition_state_dict": transition.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

            start_epoch, best_loss = train_world_model.load_training_state(
                checkpoint_path, encoder, transition, optimizer, torch.device("cpu")
            )

        self.assertEqual(start_epoch, 7)
        self.assertEqual(best_loss, 0.125)


class HarderEnvCommandTests(unittest.TestCase):
    def test_build_python_command_uses_active_interpreter(self):
        command = train_harder_env.build_python_command(
            "training/collect_trajectories.py",
            "--env",
            "MiniGrid-FourRooms-v0",
        )

        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], "training/collect_trajectories.py")
        self.assertEqual(command[2:], ["--env", "MiniGrid-FourRooms-v0"])


if __name__ == "__main__":
    unittest.main()
