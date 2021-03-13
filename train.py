from pt_datasets import load_dataset, create_dataloader


train_data, test_data = load_dataset("wdbc")
train_loader = create_dataloader(train_data, batch_size=32, num_workers=4)
