import torch.nn as nn
import torch
import tqdm


save_model_path = 'D:/5G_python/prach_ai/prach_preamble_detection_AI/weights'

model_name = 'vgg11'
weights_name = model_name + '_corr_gaf_data.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Testing on {device}...\n")

criterion = nn.CrossEntropyLoss(reduction='sum')

model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=False).to(device)

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
print(' Average Accuracy: ', avg_acc)