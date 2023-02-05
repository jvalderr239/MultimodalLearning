""" NN Architecture for MultiModal Model """

import copy
import logging
import time

import albumentations as A
import numpy as np
import torch
import torchvision.models as vismodels
from albumentations.pytorch import ToTensorV2
from fastai.vision.all import TimeDistributed
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

import transforms

# Create a custom logger
log = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("file.log")
i_handler = logging.FileHandler("results.log")
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.ERROR)
i_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
i_handler.setFormatter(c_format)
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

drive_path = "./drive/MyDrive/Multimodal/"
model_path = drive_path + "Models/"
data_path = drive_path + "Data/"
scripts_path = drive_path + "Scripts/"


def format_inputs(inputs, model_name):
    """Format dataset inputs"""
    transform = transforms.generate_transform("val")
    t_t = transform[model_name]
    samples = []
    if "visual" in model_name:
        for batch_sample in inputs:
            batch = []
            for window in batch_sample:
                sample = t_t(image=(np.array(window)))["image"]
                batch.append(sample)

            samples.append(torch.stack(batch))
    else:  # audio
        for batch_sample in inputs:
            batch = []
            for window in batch_sample:
                sample = t_t(image=window.numpy())["image"]

                batch.append(sample)
            samples.append(torch.stack(batch))
    return torch.stack(samples)


def test_model(model_name, testloader):
    """Test Individual Model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alexnet = vismodels.alexnet(pretrained=True)
    alexnet.classifier[6] = torch.nn.Identity()
    model = torch.nn.Sequential(
        TimeDistributed(alexnet),
        torch.nn.Flatten(),
        torch.nn.AvgPool1d(3),
        torch.nn.Linear((6 * 4096) // 3, 8),
        torch.nn.Softmax(dim=1),
    )

    model.load_state_dict(torch.load(model_path + model_name + "_model.pth"))
    model.cuda()

    correct, total = 0, 0
    f_1, recall, prec = 0, 0, 0
    with torch.no_grad():
        model.eval()
        for audio_inputs, visual_inputs, labels in testloader["test"]:
            if "audio" in model_name:
                inputs = format(audio_inputs, model_name)
            else:
                inputs = format(visual_inputs, model_name)
            inputs = inputs.to(device, non_blocking=True)
            labels = (labels.float()).to(device, non_blocking=True)

            outputs = model(inputs)

            # convert model output to one hot encoding
            _, preds = torch.max(outputs, 1)
            _, targets = torch.max(labels, 1)

            total += labels.size(0)
            f_1 += f1_score(
                targets.cpu().data.numpy(),
                preds.cpu().data.numpy(),
                average="macro",
                labels=np.unique(preds.cpu().data.numpy()),
            ) * labels.size(0)
            recall += recall_score(
                targets.cpu().data.numpy(),
                preds.cpu().data.numpy(),
                average="macro",
                labels=np.unique(preds.cpu().data.numpy()),
            ) * labels.size(0)
            prec += precision_score(
                targets.cpu().data.numpy(),
                preds.cpu().data.numpy(),
                average="macro",
                labels=np.unique(preds.cpu().data.numpy()),
            ) * labels.size(0)
            correct += torch.sum(preds == targets)

    log.INFO(
        f"Precision_score of the network on the {total} test images: {100 * prec/total}"
    )
    log.INFO(
        f"Recall_score of the network on the {total} test images: {100 * recall/total}"
    )
    log.INFO(f"F1_score of the network on the {total} test images: {100 * f_1/total}")
    log.INFO(
        f"Accuracy of the network on the {total} test images: {100 * (correct/total)}"
    )


class AudioVisualConcat(torch.nn.Module):
    """Fusion Network for MultiModal Model"""

    def __init__(
        self,
        num_classes,
        loss_fn,
        audio_module,
        vision_module,
        audio_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
    ):
        super(AudioVisualConcat, self).__init__()
        # for the input audio and visual pre trained networks
        # we want to remove the last layer
        self.audio_module = audio_module
        self.vision_module = vision_module
        self.fusion = torch.nn.Linear(
            in_features=(audio_feature_dim + vision_feature_dim),
            out_features=fusion_output_size,
        )
        self.f_c = torch.nn.Linear(
            in_features=fusion_output_size, out_features=num_classes
        )
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p, inplace=False)

    def forward(self, audio, image, label=None):
        """
        Forward pass for training
        """
        # activation functions
        audio_activation = torch.nn.LeakyReLU(inplace=False)
        image_activation = torch.nn.LeakyReLU(inplace=False)
        fusion_activation = torch.nn.AvgPool1d(kernel_size=1, stride=1)

        # fuse audio visual
        audio_features = audio_activation(self.audio_module(audio.clone()))
        image_features = image_activation(self.vision_module(image.clone()))
        combined = torch.cat([audio_features.clone(), image_features.clone()], dim=1)

        # fusion network
        fused_features = fusion_activation(self.fusion(combined.clone()))
        fused = self.dropout(fused_features.clone())
        logits = self.fc(fused.clone())
        pred = torch.nn.functional.softmax(logits.clone(), dim=1)
        loss_ = (
            self.loss_fn(pred.clone(), label.clone()) if label is not None else label
        )
        return (pred, loss_)


class AudioVisualModel(torch.nn.Module):
    """Multimodal Modal"""

    def __init__(self, device, pickle_file_path):
        super(AudioVisualModel, self).__init__()
        self.num_classes = 8
        self.audio_feature_dim = (6 * 4096) // 3
        self.vision_feature_dim = (6 * 4096) // 3
        self.fusion_output_size = 1024
        self.dropout_rate = 0.3
        self.model_path = model_path
        self.device = device
        self.dataloader = torch.load(
            pickle_file_path, pickle_module=pickle_file_path, map_location=self.device
        )
        self.train_transform = transforms.generate_transform("train")
        self.val_transform = transforms.generate_transform("val")
        self.model = self.build_multimodal_model()

    def forward(self, audio, image, label=None):
        """Forward Pass"""
        return self.model(audio, image, label)

    def load_model(self, filename):
        """Load auxiliary models for build"""
        alexnet = vismodels.alexnet(pretrained=True)
        alexnet.classifier[6] = torch.nn.Identity()
        model = torch.nn.Sequential(
            TimeDistributed(alexnet),
            torch.nn.Flatten(),
            torch.nn.AvgPool1d(3),
            torch.nn.Linear((6 * 4096) // 3, 8),
            torch.nn.Softmax(dim=1),
        )
        model.load_state_dict(
            torch.load(self.model_path + filename, map_location=self.device)
        )
        multimodal = model[:-2]  # remove softmax and linear layer
        for param in list(model.children())[:-2]:
            param.requires_grad = False
        return multimodal

    def build_multimodal_model(self):
        """Build Multimodal Architecture"""
        log.INFO(f"Building audio model")
        # load pre trained audio network
        audio_module = self.load_model("audio_model.pth")
        log.INFO(f"Building visual model")
        # load pre trained visual module
        vision_module = self.load_model("visual_model.pth")
        log.INFO(f"Building multimodal model")
        return AudioVisualConcat(
            num_classes=self.num_classes,
            loss_fn=torch.nn.CrossEntropyLoss(),
            audio_module=audio_module,
            vision_module=vision_module,
            audio_feature_dim=self.audio_feature_dim,
            vision_feature_dim=self.vision_feature_dim,
            fusion_output_size=self.fusion_output_size,
            dropout_p=self.dropout_rate,
        )

    def _format_inputs(self, inputs, name, phase):
        tt = (
            self.train_transform[name] if "train" in phase else self.val_transform[name]
        )
        samples = []
        if "visual" in name:
            for batch_sample in inputs:
                batch = []
                for window in batch_sample:
                    sample = tt(image=(np.array(window.cpu().data)))["image"]
                    batch.append(sample)
                samples.append(torch.stack(batch))
        else:  # audio
            for batch_sample in inputs:
                batch = []
                for window in batch_sample:
                    sample = tt(image=window.cpu().data.numpy())["image"]
                    batch.append(sample)
                samples.append(torch.stack(batch))
        return torch.stack(samples)

    def train_model(
        self,
        criterion=torch.nn.CrossEntropyLoss(),
        num_epochs=25,
        _path=data_path,
    ):
        """Train Model"""
        name = "multimodal"
        model = self.model
        model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        log.INFO(f"Running {self.device}")
        since = time.time()
        val_acc_history = []
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_epoch = 1
        torch.autograd.set_detect_anomaly(True)

        for epoch in tqdm(range(1, num_epochs + 1)):
            log.INFO("Epoch {}/{}".format(epoch, num_epochs))
            log.INFO("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                log.INFO(f"Current mode: {phase}")
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                for audio_inputs, visual_inputs, labels in self.dataloader[phase]:
                    audio_inputs = self._format_inputs(audio_inputs, "audio", phase)
                    visual_inputs = self._format_inputs(visual_inputs, "visual", phase)
                    # inputs = torch.nn.functional.normalize(inputs, p=2.0, dim=1)
                    audio_inputs = audio_inputs.to(self.device, non_blocking=True)
                    visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                    labels = (labels.float()).to(self.device, non_blocking=True)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        preds, loss = model.forward(audio_inputs, visual_inputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward(retain_graph=True)
                            optimizer.step()

                    # statistics# convert model output to one hot encoding
                    _, maxprobs = torch.max(preds, dim=1)
                    _, targets = torch.max(labels, dim=1)

                    # statistics
                    running_loss += loss.item() * labels.size(0)
                    running_corrects += torch.sum(maxprobs == targets)

                epoch_loss = running_loss / len(self.dataloader[phase].dataset)
                epoch_acc = 100 * (
                    running_corrects.double() / len(self.dataloader[phase].dataset)
                )

                log.INFO(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                if phase == "val":
                    val_acc_history.append(epoch_acc)

                if phase == "train" and (epoch % 50 == 0 or epoch == 1):
                    path = self.model_path + name + "/"
                    filename = name + "_" + phase + "_" + "epoch_" + str(epoch) + ".pth"
                    log.INFO(f"Storing {filename}")
                    torch.save(model.state_dict(), path + filename)

        time_elapsed = time.time() - since
        log.INFO(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        log.INFO("Best val Acc: {:4f}".format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        path = self.model_path + name + "/"
        filepath = path + name + "_" + "final_" + str(best_epoch) + ".pth"
        torch.save(model.state_dict(), filepath)

        with open(path + "overall_acc.txt", "w") as acc_file:
            for item in val_acc_history:
                acc_file.write("%s\n" % item)

        return model, val_acc_history

    def test_model(self):
        """Test model and plot metrics"""
        self.model.cuda()

        correct = 0
        total = 0
        f_1, prec, recall = 0, 0, 0
        phase = "test"
        with torch.no_grad():
            self.model.eval()
            for audio_inputs, visual_inputs, labels in self.dataloader[phase]:
                audio_inputs = self._format_inputs(audio_inputs, "audio", phase)
                visual_inputs = self._format_inputs(visual_inputs, "visual", phase)
                # inputs = torch.nn.functional.normalize(inputs, p=2.0, dim=1)
                audio_inputs = audio_inputs.to(self.device, non_blocking=True)
                visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                labels = (labels.float()).to(self.device, non_blocking=True)
                preds, loss = self.model(audio_inputs, visual_inputs, labels)

                # convert model output to one hot encoding
                _, maxprobs = torch.max(preds, dim=1)
                _, targets = torch.max(labels, dim=1)
                total += labels.size(0)
                f_1 += f1_score(
                    targets.cpu().data.numpy(),
                    maxprobs.cpu().data.numpy(),
                    average="macro",
                    labels=np.unique(maxprobs.cpu().data.numpy()),
                ) * labels.size(0)
                recall += recall_score(
                    targets.cpu().data.numpy(),
                    maxprobs.cpu().data.numpy(),
                    average="macro",
                    labels=np.unique(maxprobs.cpu().data.numpy()),
                ) * labels.size(0)
                prec += precision_score(
                    targets.cpu().data.numpy(),
                    maxprobs.cpu().data.numpy(),
                    average="macro",
                    labels=np.unique(maxprobs.cpu().data.numpy()),
                ) * labels.size(0)
                correct += torch.sum(maxprobs == targets)

        log.INFO(
            f"Precision_score of the network on the {total} test images: {100 * prec/total}"
        )
        log.INFO(
            f"Recall_score of the network on the {total} test images: {100 * recall/total}"
        )
        log.INFO(
            f"F1_score of the network on the {total} test images: {100 * f_1/total}"
        )
        log.INFO(
            f"Accuracy of the network on the {total} test images: {100 * (correct/total)}"
        )
