# MinigridMiniworldTransfer

## Create Custom Feature Extractor in Stable-Baselines3

The goal of this section is to look at how we can create a custom feature extractor for the [`RecurrentActorCriticCnnPolicy`](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/21cc96cafd77a3a4347b43c374a4cd21bf08d804/sb3_contrib/common/recurrent/policies.py#L434) class in `stable-baselines3-contrib`. We can see that one of the inputs `RecurrentActorCriticCnnPolicy` takes is `features_extractor_class` which is defaulted to [`NatureCNN`](https://github.com/DLR-RM/stable-baselines3/blob/96526ed08af19f48dd762f70d06e9f32c2c63d18/stable_baselines3/common/torch_layers.py#L48). We can see that the CNN network in the [`NatureCNN`](https://github.com/DLR-RM/stable-baselines3/blob/96526ed08af19f48dd762f70d06e9f32c2c63d18/stable_baselines3/common/torch_layers.py#L48) class has the architecture

```python
self.cnn = nn.Sequential(
    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
    nn.Flatten(),
)
```

This does not work with `Minigrid` observations, which rarely has shape larger than `3x9x9`. A better option for the CNN architecture is what is shown in [`rl-starter-files`](https://github.com/lcswillems/rl-starter-files/blob/8596478c988fe780b721cea8e44c60563afec5f6/model.py#L27)

```python
self.image_conv = nn.Sequential(
    nn.Conv2d(3, 16, (2, 2)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    nn.Conv2d(16, 32, (2, 2)),
    nn.ReLU(),
    nn.Conv2d(32, 64, (2, 2)),
    nn.ReLU()
)
```