import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch import autograd


class model_generator():
    def __init__(self, train_dataloader, valid_dataloader, model, loss_fn, optimizer, num_epoch, save_path):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.save_path = save_path

    def training_loop(self):
        # training_loop
        train_size = len(self.train_dataloader.dataset)
        true_positive, false_negative, false_positive = 0, 0, 0
        running_loss = 0.0

        for batch, (X, y) in enumerate(self.train_dataloader):
            # Compute prediction and loss
            X = X.float().cuda()
            pred = self.model(X)
            loss = self.loss_fn(pred, y.cuda())

            # Backpropagation
            with autograd.detect_anomaly():
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # training loss sum
            running_loss += loss.item() * len(X)

            # calculate tp, fp, tn, fn
            loss, current = loss.item(), batch * len(X)
            tn, fp, fn, tp = confusion_matrix(y.cpu(), pred.cpu().argmax(1).numpy()).ravel()
            true_positive += tp
            false_negative += fn
            false_positive += fp

            print(
                f"step:{batch:>3d} - loss:{loss:>3f} - IoU:{tp / (tp + fn + fp):>3f} - tp:{tp:>2d} - fn:{fn:>2d} - fp:{fp:>3d} - tn:{tn:>3d} [{current:>6d}/{train_size:>6d}]")

        # print average loss value and IoU
        training_loss = running_loss / valid_size
        training_IoU = true_positive / (true_positive + false_negative + false_positive)
        print(f"Train Error: \n Train_Loss: {training_loss:>8f}, Train_IoU: {training_IoU:>7f} \n")

        return training_loss, training_IoU

    def testing_loop(self):
        # testing_loop
        valid_size = len(self.valid_dataloader.dataset)
        true_positive, false_negative, false_positive = 0, 0, 0
        running_loss = 0.0

        print('-------------------------------------Testing---------------------------------------------')
        with torch.no_grad():
            for batch, (X, y) in enumerate(self.valid_dataloader):
                # Compute prediction and loss
                X = X.float().cuda()
                pred = self.model(X)
                loss = self.loss_fn(pred, y.cuda())

                # testing loss sum
                running_loss += loss.item() * len(X)

                # calculate tp, fp, tn, fn
                loss, current = loss.item(), batch * len(X)
                tn, fp, fn, tp = confusion_matrix(y.cpu(), pred.cpu().argmax(1).numpy()).ravel()
                true_positive += tp
                false_negative += fn
                false_positive += fp

                print(
                    f"step:{batch:>3d} - loss:{loss:>3f} - IoU:{tp / (tp + fn + fp):>3f} - tp:{tp:>2d} - fn:{fn:>2d} - fp:{fp:>3d} - tn:{tn:>3d} [{current:>6d}/{valid_size:>6d}]")

        # print loss value and IoU Evaluation
        testing_loss = running_loss / valid_size
        testing_IoU = true_positive / (true_positive + false_negative + false_positive)
        print(f"Test Error: \n Val_loss: {testing_loss:>8f}, Val_IoU: {testing_IoU:>7f} \n")

        return testing_loss, testing_IoU

    def run(self):

        train_loss = []
        train_IoU = []
        valid_loss = []
        valid_IoU = []
        the_best_IoU = 0

        for epoch in range(self.num_epoch):
            print(f"Epoch {epoch + 1}:\n------------------------------------------------")

            # training_loop
            training_loss, training_IoU = self.training_loop()
            train_loss.append(training_loss)
            train_IoU.append(training_IoU)

            # testing_loop
            testing_loss, testing_IoU = self.testing_loop()
            valid_loss.append(testing_loss)
            valid_IoU.append(testing_IoU)

            # save the best model
            if testing_IoU > the_best_IoU:
                the_best_IoU = testing_IoU
                torch.save(self.model.state_dict(), self.save_path)

        print('Done!')
        return train_loss, valid_loss, train_IoU, valid_IoU
