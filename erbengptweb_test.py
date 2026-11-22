import torch  # noqa: I001
from erbengptweb import train_job, load_checkpoint, TrainingConfig


def test_checkpoint_resume_correctness(tmp_path):
    device = torch.device("cpu")

    total_steps = 100
    lr = 1e-3

    # first train a model
    path1 = str(tmp_path / "run-1")
    max_sequence = 64
    feature_count = 64
    head_count = 4
    layer_count = 2
    data_path = "data/processed/fineweb-2000"

    config1 = TrainingConfig(data_path, path1, total_steps, total_steps, 4, 8, False)
    train_job(config1, lr, device, max_sequence, feature_count, head_count, layer_count, 
              load=False, training=True, final_validation=False)

    # then train model 2 using repeated save & load
    count = 10
    steps = int(total_steps / count)
    path2 = str(tmp_path / "run-2")
    config2 = TrainingConfig(data_path, path2, total_steps, steps, 4, 8, False)
    for i in range(count):
        load = i > 0
        train_job(config2, lr, device, max_sequence, feature_count, head_count, layer_count,
                  load=load, training=True, final_validation=False)

    # load both models checkpoints
    model1, optimizer1, lr_scheduler1, metrics1 = load_checkpoint(config1, device, lr, total_steps)
    model2, optimizer2, lr_scheduler2, metrics2 = load_checkpoint(config2, device, lr, total_steps)

    # assert the 2 models have the same weight
    torch.testing.assert_close(model1.state_dict(), model2.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(optimizer1.state_dict(), optimizer2.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(lr_scheduler1.state_dict(), lr_scheduler2.state_dict(), rtol=0, atol=0)
    assert metrics1.completed_updates == metrics2.completed_updates == total_steps
    assert metrics1.processed_tokens == metrics2.processed_tokens