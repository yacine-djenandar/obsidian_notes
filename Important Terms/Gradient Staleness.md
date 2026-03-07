#gradient_staleness 

**Gradient staleness** is a problem that happens when the information used to update a model is "old" or "outdated." In distributed training, this typically occurs because of the time delay between when a worker calculates a gradient and when that gradient is finally used to update the global model.
