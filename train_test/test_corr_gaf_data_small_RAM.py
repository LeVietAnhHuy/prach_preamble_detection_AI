print(f"Start testing ...")

model = torch.hub.load('pytorch/vision:v0.10.0', model_name[model_idx], pretrained=False).to(device)
model.load_state_dict(torch.load(os.path.join(save_model_path, model_name + '_corr_gaf_data.pth')))

total_acc = []
num_test = 10
for i in tqdm(range(num_test)):
    correct, total = 0, 0
    for batch in test_dl:
        x, y_batch = [t.to(device) for t in batch]
        out = model(x)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_batch.size(0)
        correct += (preds == y_batch).sum().item()

    total_acc.append(correct / total)
avg_acc = sum(total_acc) / num_test
print('done!')
print('Average Accuracy: ', avg_acc)