# Training API

## Trainer

`ununennium.training.Trainer`

Features:
- Mixed precision training (AMP)
- Distributed Data Parallel (DDP)
- Gradle accumulation
- Experiment logging

### Usage

```python
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    callbacks=[EarlyStopping(), ModelCheckpoint()]
)
trainer.fit(train_loader, val_loader, epochs=100)
```

## Callbacks

### `EarlyStopping`

Stop training when validation metric stops improving.

### `ModelCheckpoint`

Save model weights periodically or on best metric.

### `LearningRateMonitor`

Log learning rate changes.
