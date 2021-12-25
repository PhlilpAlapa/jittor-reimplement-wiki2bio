import argparse
from preprocess import Vocab
from SeqUnit import SeqUnit
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-box', type=str)
parser.add_argument('--hidden_size', type=int,
                    default=500, help='Size of each layer')
parser.add_argument('--emb_size', type=int,
                    default=400, help='Size of embedding')
parser.add_argument('--field_size', type=int,
                    default=50, help='Size of embedding')
parser.add_argument('--pos_size', type=int,
                    default=5, help='Size of embedding')
parser.add_argument('--batch_size', type=int,
                    default=32, help='Batch size of train set')
parser.add_argument('--epoch', type=int,
                    default=50, help='Number of training epoch')
parser.add_argument('--source_vocab', type=int,
                    default=20003, help='vocabulary size')
parser.add_argument('--field_vocab', type=int,
                    default=1480, help='vocabulary size')
parser.add_argument('--position_vocab', type=int,
                    default=31, help='vocabulary size')
parser.add_argument('--target_vocab', type=int,
                    default=20003, help='vocabulary size')
parser.add_argument('--report', type=int,
                    default=1000, help='report valid results after some steps')
parser.add_argument('--learning_rate', type=float,
                    default=0.0005, help='learning rate')

parser.add_argument('--mode', type=str,
                    default='train', help='train or test')
parser.add_argument('--load', type=str,
                    default='0', help='load directory')
parser.add_argument('--dir', type=str,
                    default='processed_data', help='data set directory')
parser.add_argument('--limits', type=int,
                    default=0, help='max data set size')

parser.add_argument('--dual_attention', type=bool,
                    default=True, help='dual attention layer or normal attention')
parser.add_argument('--fgate_encoder', type=bool,
                    default=True, help='add field gate in encoder lstm')

parser.add_argument('--field', type=bool,
                    default=False, help='concat field information to word embedding')
parser.add_argument('--position', type=bool,
                    default=False, help='concat position information to word embedding')
parser.add_argument('--encoder_pos', type=bool,
                    default=True, help='position information in field-gated encoder')
parser.add_argument('--decoder_pos', type=bool,
                    default=True, help='position information in field-gated encoder')

FLAGS = parser.parse_args()

model = SeqUnit(batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size, emb_size=FLAGS.emb_size,
                field_size=FLAGS.field_size, pos_size=FLAGS.pos_size, field_vocab=FLAGS.field_vocab,
                source_vocab=FLAGS.source_vocab, position_vocab=FLAGS.position_vocab,
                target_vocab=FLAGS.target_vocab, field_concat=FLAGS.field, position_concat=FLAGS.position,
                fgate_enc=FLAGS.fgate_encoder, dual_att=FLAGS.dual_attention, decoder_add_pos=FLAGS.decoder_pos,
                encoder_add_pos=FLAGS.encoder_pos)
model.load('C:/Users/11202/Desktop/AI/model.pkl')

vocab = Vocab()


def preprocess(field_values):
    text = []
    val = []
    lab = []
    pos = []
    rpos = []

    text_len = 0
    for field in field_values:
        values = field_values[field].split(' ')

        length = len(values)
        text_len += length

        for i in range(length):
            text.append(values[i])
            val.append(vocab.word2id(values[i]))
            lab.append(vocab.key2id(field))
            pos.append(i + 1)
            rpos.append(length - i)

    return {'enc_in': [val], 'enc_fd': [lab], 'enc_pos': [pos], 'enc_rpos': [rpos], 'enc_len': [text_len]}, text


def generate(field_values):
    x, text = preprocess(field_values)
    predictions, atts = model(x, False)
    atts = np.squeeze(atts.data)

    summary = list(predictions[0])
    if 2 in summary:
        summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
    real_sum, unk_sum, mask_sum = [], [], []
    for tk, tid in enumerate(summary):
        if tid == 3:
            sub = text[np.argmax(atts[tk, : len(text)])]
            real_sum.append(sub)
            mask_sum.append("**" + str(sub) + "**")
        else:
            real_sum.append(vocab.id2word(tid))
            mask_sum.append(vocab.id2word(tid))

    return ' '.join(real_sum)


field_values = FLAGS.box.split('*')[:-1]
key_values = {}
for field_value in field_values:
    field, value = field_value.split(':')
    key_values[field] = value.replace("~", " ")

# name:dale~raoul*birth_date:16~august~1956*birth_place:missoula~,~montana*occupation:actress*years_active:1986-present*spouse:ray~thompson*article_title:dale~raoul
# name:George~Mikell*birth_name:Jurgis~Mikelaitis*birth_date:4~April~1929*birth_place:Bildeniai~,~Lithuania*nationality:Lithuanian~,~Australian*occupation:Actor~,~writer*

# field_values = {
#     'name': 'George Mikell',
#     'birth_name': 'Jurgis Mikelaitis',
#     'birth_date': '4 April 1929 ( age 88 )',
#     'birth_place': 'Bildeniai , Lithuania',
#     'nationality': 'Lithuanian , Australian',
#     'occupation': 'Actor , writer'
# }

result = generate(key_values)
print("generate:", result)
