import torch  # noqa: I001
from erbengptweb import train_job, load_checkpoint


def test_checkpoint_resume_correctness(tmp_path):
    device = torch.device("cpu")

    total_steps = 100
    lr = 1e-3

    # first train a model
    ckpt1 = tmp_path / "ckpt1"
    train_job(total_steps, lr, device, load=False, skip_training=False, model_ckpt_path=ckpt1)

    # then train model 2 using repeated save & load
    count = 10
    steps = int(total_steps / count)
    ckpt2 = tmp_path / "ckpt2"
    for i in range(count):
        load = i > 0
        train_job(steps, lr, device, load=load, skip_training=False, model_ckpt_path=ckpt2)


    # load both models checkpoints
    model1, optimizer1, lr_scheduler1, completed_steps1 = load_checkpoint(ckpt1, device, lr)
    model2, optimizer2, lr_scheduler2, completed_steps2 = load_checkpoint(ckpt2, device, lr)

    # assert the 2 models have the same weight
    torch.testing.assert_close(model1.state_dict(), model2.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(optimizer1.state_dict(), optimizer2.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(lr_scheduler1.state_dict(), lr_scheduler2.state_dict(), rtol=0, atol=0)
    assert completed_steps1 == completed_steps2 == total_steps