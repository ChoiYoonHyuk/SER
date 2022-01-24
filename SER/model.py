import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Function
from tqdm import tqdm

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
batch_size = 32


# Define GRL for common feature extraction
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.lambda_ = 1
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = 1
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


# Define model
class SER(nn.Module):
    def __init__(self):
        super(SER, self).__init__()
        # Num of CNN filter, CNN filter size 5x100
        self.filters_num = 100
        self.kernel_size = 5
        # Word embedding dimension
        self.word_dim = 100
        # Loss for siamese encoder
        self.dist = nn.MSELoss()

        self.s_user_feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim)),
            nn.BatchNorm2d(self.filters_num),
            nn.Sigmoid(),
            nn.MaxPool2d((496, 1)),
            nn.Dropout(),
        )

        self.s_item_feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim)),
            nn.BatchNorm2d(self.filters_num),
            nn.Sigmoid(),
            nn.MaxPool2d((496, 1)),
            nn.Dropout(),
        )

        self.t_user_feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim)),
            nn.BatchNorm2d(self.filters_num),
            nn.Sigmoid(),
            nn.MaxPool2d((496, 1)),
            nn.Dropout(),
        )

        self.t_item_feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim)),
            nn.BatchNorm2d(self.filters_num),
            nn.Sigmoid(),
            nn.MaxPool2d((496, 1)),
            nn.Dropout(),
        )

        self.c_user_feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim)),
            nn.BatchNorm2d(self.filters_num),
            nn.Sigmoid(),
            nn.MaxPool2d((496, 1)),
            nn.Dropout(),
        )

        self.c_item_feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim)),
            nn.BatchNorm2d(self.filters_num),
            nn.Sigmoid(),
            nn.MaxPool2d((496, 1)),
            nn.Dropout(),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(200, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
        )

        self.s_encoder = nn.Sequential(
            nn.Linear(200, 200)
        )

        self.s_classifier = nn.Sequential(
            nn.Linear(200, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
        )

        self.t_encoder = nn.Sequential(
            nn.Linear(200, 200)
        )

        self.t_classifier = nn.Sequential(
            nn.Linear(200, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
        )

        self.reset_para()

    def reset_para(self):
        for cnn in [self.s_user_feature_extractor[0], self.s_item_feature_extractor[0]]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        for cnn in [self.t_user_feature_extractor[0], self.t_item_feature_extractor[0]]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        for cnn in [self.c_user_feature_extractor[0], self.c_item_feature_extractor[0]]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        for fc in [self.s_classifier[0]]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

        for fc in [self.t_classifier[0]]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

    def forward(self, user, item, ans, label):
        # Source individual review FE
        s_u_ans_fea = self.s_user_feature_extractor(ans).squeeze(2).squeeze(2)
        c_u_ans_fea = self.c_user_feature_extractor(ans).squeeze(2).squeeze(2)
        s_u_ans_fea = (s_u_ans_fea + c_u_ans_fea) / 2

        s_i_ans_fea = self.s_item_feature_extractor(ans).squeeze(2).squeeze(2)
        c_i_ans_fea = self.c_item_feature_extractor(ans).squeeze(2).squeeze(2)
        s_i_ans_fea = (s_i_ans_fea + c_i_ans_fea) / 2

        s_ans_fea = torch.cat((s_u_ans_fea, s_i_ans_fea), 1).squeeze(1)

        # Label of source individual review
        s_cls_out = self.s_classifier(s_ans_fea)

        # Output is [Source | Target] --> Masking target output for loss calculation
        masking = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).view(
            batch_size * 2, -1).to(device)
        s_ans_out, s_label = torch.mul(
            s_cls_out, masking), torch.mul(label, masking)

        # Source aggregated reviews FE
        s_u_fea = self.s_user_feature_extractor(user).squeeze(2).squeeze(2)
        s_i_fea = self.s_item_feature_extractor(item).squeeze(2).squeeze(2)

        s_c_u_fea = self.c_user_feature_extractor(user).squeeze(2).squeeze(2)
        s_c_i_fea = self.c_item_feature_extractor(item).squeeze(2).squeeze(2)

        s_u_fea = (s_u_fea + s_c_u_fea) / 2
        s_i_fea = (s_i_fea + s_c_i_fea) / 2

        s_fea = torch.cat((s_u_fea, s_i_fea), 1).squeeze(1)

        # Passing through encoder for aggregated review embedding
        s_fea = self.s_encoder(s_fea)

        s_cls_out = self.s_classifier(s_fea)
        s_out = torch.mul(s_cls_out, masking)

        # Distance between individual review & aggregated review
        s_dist = self.dist(torch.mul(s_ans_fea, masking),
                           torch.mul(s_fea, masking))

        # Same for target domain
        t_u_ans_fea = self.t_user_feature_extractor(ans).squeeze(2).squeeze(2)
        c_u_ans_fea = self.c_user_feature_extractor(ans).squeeze(2).squeeze(2)
        t_u_ans_fea = (t_u_ans_fea + c_u_ans_fea) / 2

        t_i_ans_fea = self.t_item_feature_extractor(ans).squeeze(2).squeeze(2)
        c_i_ans_fea = self.c_item_feature_extractor(ans).squeeze(2).squeeze(2)
        t_i_ans_fea = (t_i_ans_fea + c_i_ans_fea) / 2

        t_ans_fea = torch.cat((t_u_ans_fea, t_i_ans_fea), 1).squeeze(1)

        t_cls_out = self.t_classifier(t_ans_fea)

        masking = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)]).view(
            batch_size * 2, -1).to(device)
        t_ans_out, t_label = torch.mul(
            t_cls_out, masking), torch.mul(label, masking)

        # Target classification loss
        t_u_fea = self.t_user_feature_extractor(user).squeeze(2).squeeze(2)
        t_i_fea = self.t_item_feature_extractor(item).squeeze(2).squeeze(2)

        t_c_u_fea = self.c_user_feature_extractor(user).squeeze(2).squeeze(2)
        t_c_i_fea = self.c_item_feature_extractor(item).squeeze(2).squeeze(2)

        t_u_fea = (t_u_fea + t_c_u_fea) / 2
        t_i_fea = (t_i_fea + t_c_i_fea) / 2

        t_fea = torch.cat((t_u_fea, t_i_fea), 1).squeeze(1)

        t_fea = self.t_encoder(t_fea)

        t_cls_out = self.t_classifier(t_fea)
        t_out = torch.mul(t_cls_out, masking)

        t_dist = self.dist(torch.mul(t_ans_fea, masking),
                           torch.mul(t_fea, masking))

        # Discriminator label
        s_domain_specific = torch.zeros(batch_size).to(device)
        t_domain_specific = torch.ones(batch_size).to(device)

        # Common source discriminator loss
        s_c_d_fea = torch.cat((s_c_u_fea, s_c_i_fea), 1)
        s_c_d_fea = GradientReversalFunction.apply(s_c_d_fea)
        s_c_d_fea = self.discriminator(s_c_d_fea).squeeze(1)[0:batch_size]
        s_c_domain_loss = F.binary_cross_entropy_with_logits(
            s_c_d_fea, s_domain_specific)

        # Common target discriminator loss
        t_c_d_fea = torch.cat((t_c_u_fea, t_c_i_fea), 1)
        t_c_d_fea = GradientReversalFunction.apply(t_c_d_fea)
        t_c_d_fea = self.discriminator(t_c_d_fea).squeeze(1)[
            batch_size:batch_size * 2]
        t_c_domain_loss = F.binary_cross_entropy_with_logits(
            t_c_d_fea, t_domain_specific)

        domain_common_loss = (s_c_domain_loss + t_c_domain_loss) / 2

        # Source specific discriminator loss
        s_d_fea = torch.cat((s_u_fea, s_i_fea), 1)
        s_d_fea = self.discriminator(s_d_fea).squeeze(1)[0:batch_size]

        # Target specific discriminator loss
        t_d_fea = torch.cat((t_u_fea, t_i_fea), 1)
        t_d_fea = self.discriminator(t_d_fea).squeeze(1)[
            batch_size:batch_size * 2]

        s_domain_specific = torch.zeros(batch_size).to(device)
        s_domain_loss = F.binary_cross_entropy_with_logits(
            s_d_fea, s_domain_specific)
        t_domain_specific = torch.ones(batch_size).to(device)
        t_domain_loss = F.binary_cross_entropy_with_logits(
            t_d_fea, t_domain_specific)
        domain_specific_loss = (s_domain_loss + t_domain_loss) / 2

        return s_ans_out, s_out, s_label, s_dist, t_ans_out, t_out, t_label, t_dist, domain_common_loss, domain_specific_loss


# Clean strings for reviews
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)

    return string.strip().lower()


# Review embedding layer
def pre_processing(s_data, s_dict, t_data, t_dict, w_embed, valid_idx):
    # Return embedded vector [user, item, rev_ans, rat]
    u_embed, i_embed, ans_embed, label = [], [], [], []
    limit = 500

    for idx in range(batch_size):
        u, i, rat = s_data[0][idx], s_data[1][idx], s_data[2][idx]

        u_rev, i_rev, ans_rev = [], [], []

        reviews = s_dict[u]
        for review in reviews:
            if review[0] != i:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        u_rev.append(rev)
                        if len(u_rev) > limit:
                            break
                    except KeyError:
                        continue

        reviews = s_dict[i]
        for review in reviews:
            if review[0] != u:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        i_rev.append(rev)
                        if len(i_rev) > limit:
                            break
                    except KeyError:
                        continue

        reviews = s_dict[u]
        for review in reviews:
            if review[0] == i:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        ans_rev.append(rev)
                        if len(ans_rev) > limit:
                            break
                    except KeyError:
                        continue

        if len(u_rev) > limit:
            u_rev = u_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(u_rev)
            for p in range(pend):
                u_rev.append(lis)

        if len(i_rev) > limit:
            i_rev = i_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(i_rev)
            for p in range(pend):
                i_rev.append(lis)

        if len(ans_rev) > limit:
            ans_rev = ans_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(ans_rev)
            for p in range(pend):
                ans_rev.append(lis)

        u_embed.append(u_rev)
        i_embed.append(i_rev)
        ans_embed.append(ans_rev)
        label.append([rat])

    if valid_idx:
        u_embed = torch.tensor(u_embed, requires_grad=True).view(
            batch_size, 1, 500, 100).to(device)
        i_embed = torch.tensor(i_embed, requires_grad=True).view(
            batch_size, 1, 500, 100).to(device)
        ans_embed = torch.tensor(ans_embed, requires_grad=True).view(
            batch_size, 1, 500, 100).to(device)
        label = torch.FloatTensor(label).to(device)

        return u_embed, i_embed, ans_embed, label

    for idx in range(batch_size):
        u, i, rat = t_data[0][idx], t_data[1][idx], t_data[2][idx]

        u_rev, i_rev, ans_rev = [], [], []

        reviews = t_dict[u]
        for review in reviews:
            if review[0] != i:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        u_rev.append(rev)
                        if len(u_rev) > limit:
                            break
                    except KeyError:
                        continue

        reviews = t_dict[i]
        for review in reviews:
            if review[0] != u:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        i_rev.append(rev)
                        if len(i_rev) > limit:
                            break
                    except KeyError:
                        continue

        reviews = t_dict[u]
        for review in reviews:
            if review[0] == i:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        ans_rev.append(rev)
                        if len(ans_rev) > limit:
                            break
                    except KeyError:
                        continue

        if len(u_rev) > limit:
            u_rev = u_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(u_rev)
            for p in range(pend):
                u_rev.append(lis)

        if len(i_rev) > limit:
            i_rev = i_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(i_rev)
            for p in range(pend):
                i_rev.append(lis)

        if len(ans_rev) > limit:
            ans_rev = ans_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(ans_rev)
            for p in range(pend):
                ans_rev.append(lis)

        u_embed.append(u_rev)
        i_embed.append(i_rev)
        ans_embed.append(ans_rev)
        label.append([rat])

    u_embed = torch.tensor(u_embed, requires_grad=True).view(
        batch_size * 2, 1, 500, 100).to(device)
    i_embed = torch.tensor(i_embed, requires_grad=True).view(
        batch_size * 2, 1, 500, 100).to(device)
    ans_embed = torch.tensor(ans_embed, requires_grad=True).view(
        batch_size * 2, 1, 500, 100).to(device)
    label = torch.FloatTensor(label).to(device)

    return u_embed, i_embed, ans_embed, label


# Validation & Inference
def valid(v_data, t_data, t_dict, w_embed, save, write_file):
    model = SER()
    model.load_state_dict(torch.load(save, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.MSELoss()

    t_user_feature_extractor = model.t_user_feature_extractor
    t_item_feature_extractor = model.t_item_feature_extractor
    t_encoder = model.t_encoder
    t_clf = model.t_classifier

    c_user_feature_extractor = model.c_user_feature_extractor
    c_item_feature_extractor = model.c_item_feature_extractor

    v_batch = DataLoader(v_data, batch_size=batch_size,
                         shuffle=True, num_workers=2)
    v_loss, idx = 0, 0

    for v_data in tqdm(v_batch, leave=False):
        if len(v_data[0]) != batch_size:
            continue
        u_embed, i_embed, ans_embed, label = pre_processing(
            v_data, t_dict, v_data, t_dict, w_embed, 1)

        with torch.no_grad():
            # Target rating encoder
            c_u_fea = c_user_feature_extractor(u_embed).squeeze(2).squeeze(2)
            c_i_fea = c_item_feature_extractor(i_embed).squeeze(2).squeeze(2)

            t_u_fea = t_user_feature_extractor(u_embed).squeeze(2).squeeze(2)
            t_i_fea = t_item_feature_extractor(i_embed).squeeze(2).squeeze(2)

            u_fea, i_fea = (c_u_fea + t_u_fea) / 2, (c_i_fea + t_i_fea) / 2

            t_fea = t_encoder(torch.cat((u_fea, i_fea), 1).squeeze(1))

            t_out = t_clf(t_fea)

            v_loss += criterion(t_out, label)
        idx += 1
    v_loss = v_loss / idx

    t_batch = DataLoader(t_data, batch_size=batch_size,
                         shuffle=True, num_workers=2)
    t_loss, idx = 0, 0

    for t_data in tqdm(t_batch, leave=False):
        if len(t_data[0]) != batch_size:
            continue
        u_embed, i_embed, ans_embed, label = pre_processing(
            t_data, t_dict, t_data, t_dict, w_embed, 1)

        with torch.no_grad():
            # Target rating encoder
            c_u_fea = c_user_feature_extractor(u_embed).squeeze(2).squeeze(2)
            c_i_fea = c_item_feature_extractor(i_embed).squeeze(2).squeeze(2)

            t_u_fea = t_user_feature_extractor(u_embed).squeeze(2).squeeze(2)
            t_i_fea = t_item_feature_extractor(i_embed).squeeze(2).squeeze(2)

            u_fea, i_fea = (c_u_fea + t_u_fea) / 2, (c_i_fea + t_i_fea) / 2

            t_fea = t_encoder(torch.cat((u_fea, i_fea), 1).squeeze(1))

            t_out = t_clf(t_fea)

            t_loss += criterion(t_out, label)
        idx += 1

    t_loss = t_loss / idx

    print('Loss: %.4f %.4f' % (v_loss, t_loss))

    w = open(write_file, 'a')
    w.write('%.6f %.6f\n' % (v_loss, t_loss))


# Training
def learning(s_data, s_dict, t_data, t_dict, w_embed, save, idx):
    # Model
    print('Start Training ... \n')
    enc_loss_ratio, domain_loss_ratio = 0.05, 0.1
    model = SER()
    # After 1 epoch, load trained parameters
    if idx == 1:
        model.load_state_dict(torch.load(save, map_location=device))
    model.to(device)
    model.train()

    criterion = nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Make batch
    batch_size = 32
    s_batch = DataLoader(s_data, batch_size=batch_size,
                         shuffle=True, num_workers=2)
    t_batch = DataLoader(t_data, batch_size=batch_size,
                         shuffle=True, num_workers=2)

    batch_data, zip_size = zip(s_batch, t_batch), min(
        len(s_batch), len(t_batch))

    for source_x, target_x in tqdm(batch_data, leave=False, total=zip_size):
        # Pre processing
        if len(source_x[0]) != batch_size or len(target_x[0]) != batch_size:
            continue

        # Get embedding of user and item reviews
        u_embed, i_embed, ans_embed, label = pre_processing(
            source_x, s_dict, target_x, t_dict, w_embed, 0)

        s_ans_out, s_out, s_label, s_dist, t_ans_out, t_out, t_label, t_dist, \
            c_domain_loss, domain_loss = model(
                u_embed, i_embed, ans_embed, label)

        # Loss
        s_ans_loss, s_loss = criterion(
            s_ans_out, s_label) * 2, criterion(s_out, s_label) * 2
        t_ans_loss, t_loss = criterion(
            t_ans_out, t_label) * 2, criterion(t_out, t_label) * 2

        # Train
        loss_func = (s_loss + t_loss + s_ans_loss + t_ans_loss) / 2 + \
                    (s_dist + t_dist) * enc_loss_ratio + \
            (c_domain_loss + domain_loss) * domain_loss_ratio

        optim.zero_grad()
        loss_func.backward()
        optim.step()

        torch.save(model.state_dict(), save)

        print('Prediction Loss / Encoder Loss / Domain Loss: %.2f %.2f %.2f %.2f %.2f %.2f' %
              (s_loss, t_loss, s_dist, t_dist, c_domain_loss, domain_loss))
