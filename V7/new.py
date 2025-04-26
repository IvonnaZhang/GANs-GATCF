def valid_one_epoch(self, dataModule):
    val_loss = 0.
    preds = []
    reals = []
    for valid_Batch in tqdm(dataModule.valid_loader):
        inputs, value = valid_Batch
        inputs = inputs.to(self.args.device)
        value = value.to(self.args.device)
        pred = self.forward(inputs)
        val_loss += self.loss_function(pred, value.long())
        if self.args.classification:
            pred = torch.max(pred, 1)[1]  # 获取预测的类别标签
        preds.append(pred)
        reals.append(value)
    reals = torch.cat(reals, dim=0)
    preds = torch.cat(preds, dim=0)
    self.scheduler.step(val_loss)
    valid_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value, self.args)
    return valid_error


def test_one_epoch(self, dataModule):
    preds = []
    reals = []
    for test_Batch in tqdm(dataModule.test_loader):
        inputs, value = test_Batch
        inputs = inputs.to(self.args.device)
        value = value.to(self.args.device)
        pred = self.forward(inputs)
        if self.args.classification:
            pred = torch.max(pred, 1)[1]  # 获取预测的类别标签
        preds.append(pred)
        reals.append(value)
    reals = torch.cat(reals, dim=0)
    preds = torch.cat(preds, dim=0)
    test_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value, self.args)
    return test_error