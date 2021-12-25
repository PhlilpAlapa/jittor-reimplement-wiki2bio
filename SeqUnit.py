import jittor as jt
from jittor import Module, nn
from AttentionUnit import AttentionWrapper
from dualAttentionUnit import dualAttentionWrapper
from LstmUnit import LstmUnit
from fgateLstmUnit import fgateLstmUnit
from OutputUnit import OutputUnit


class SeqUnit(Module):
    def __init__(self, batch_size, hidden_size, emb_size, field_size, pos_size, source_vocab, field_vocab,
                 position_vocab, target_vocab, field_concat, position_concat, fgate_enc, dual_att,
                 encoder_add_pos, decoder_add_pos, start_token=2, stop_token=2, max_length=150):
        '''
        batch_size, hidden_size, emb_size, field_size, pos_size: size of batch; hidden layer; word/field/position embedding
        source_vocab, target_vocab, field_vocab, position_vocab: vocabulary size of encoder words; decoder words; field types; position
        field_concat, position_concat: bool values, whether concat field/position embedding to word embedding for encoder inputs or not
        fgate_enc, dual_att: bool values, whether use field-gating / dual attention or not
        encoder_add_pos, decoder_add_pos: bool values, whether add position embedding to field-gating encoder / decoder with dual attention or not
        '''
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.field_size = field_size
        self.pos_size = pos_size
        self.uni_size = emb_size if not field_concat else emb_size+field_size
        self.uni_size = self.uni_size if not position_concat else self.uni_size+2*pos_size
        self.field_encoder_size = field_size if not encoder_add_pos else field_size+2*pos_size
        self.field_attention_size = field_size if not decoder_add_pos else field_size+2*pos_size
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.field_vocab = field_vocab
        self.position_vocab = position_vocab
        self.start_token = start_token
        self.stop_token = stop_token
        self.max_length = max_length
        self.field_concat = field_concat
        self.position_concat = position_concat
        self.fgate_enc = fgate_enc
        self.dual_att = dual_att
        self.encoder_add_pos = encoder_add_pos
        self.decoder_add_pos = decoder_add_pos

        self.embedding = nn.Embedding(self.source_vocab, self.emb_size)

        if self.fgate_enc:
             self.enc_lstm = fgateLstmUnit(
                 self.hidden_size, self.uni_size, self.field_encoder_size)
        else:
            self.enc_lstm = LstmUnit(self.hidden_size, self.uni_size)

        self.dec_lstm = LstmUnit(self.hidden_size, self.emb_size)
        self.dec_out = OutputUnit(self.hidden_size, self.target_vocab)

        if self.dual_att:
            self.att_layer = dualAttentionWrapper(
                self.hidden_size, self.hidden_size, self.field_attention_size)
        else:
            self.att_layer = AttentionWrapper(self.hidden_size, self.hidden_size)

        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
            self.fembedding = nn.Embedding(self.field_vocab, self.field_size)
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            self.pembedding = nn.Embedding(self.position_vocab, self.pos_size)
            self.rembedding = nn.Embedding(self.position_vocab, self.pos_size)

    def execute(self, x, is_train):
        encoder_input = jt.array(x['enc_in'])
        encoder_field = jt.array(x['enc_fd'])
        encoder_pos = jt.array(x['enc_pos'])
        encoder_rpos = jt.array(x['enc_rpos'])
        encoder_len = jt.array(x['enc_len'])

        if is_train:
            decoder_input = jt.array(x['dec_in'])
            decoder_len = jt.array(x['dec_len'])
            decoder_output = jt.array(x['dec_out'])

        # ======================================== embeddings ======================================== #
        
        encoder_embed = self.embedding(encoder_input)
        if is_train:
            decoder_embed = self.embedding(decoder_input)

        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
            field_embed = self.fembedding(encoder_field)
            field_pos_embed = field_embed
            if self.field_concat:
                encoder_embed = jt.concat([encoder_embed, field_embed], 2)
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            pos_embed = self.pembedding(encoder_pos)
            rpos_embed = self.rembedding(encoder_rpos)
            if self.position_concat:
                encoder_embed = jt.concat([encoder_embed, pos_embed, rpos_embed], 2)
                field_pos_embed = jt.concat([field_embed, pos_embed, rpos_embed], 2)
            elif self.encoder_add_pos or self.decoder_add_pos:
                field_pos_embed = jt.concat([field_embed, pos_embed, rpos_embed], 2)

        # ======================================== encoder ======================================== #
        
        def encoder(inputs, inputs_len):
            batch_size = encoder_input.shape[0]
            hidden_size = self.hidden_size
            print(inputs_len)

            h0 = (jt.zeros([batch_size, hidden_size]),
                  jt.zeros([batch_size, hidden_size]))
            f0 = jt.zeros([batch_size], dtype=jt.int32)

            inputs_ta = jt.transpose(inputs, [1, 0, 2])
            emit_ta = []

            t, x_t, s_t, finished = 0, inputs_ta[0], h0, f0
            while(jt.reduce_add(finished) != f0.shape[0]):
                o_t, s_nt = self.enc_lstm(x_t, s_t, finished)
                emit_ta.append(o_t)
                finished = jt.greater_equal(t+1, inputs_len)
                print(finished)
                if(jt.reduce_logical_and(finished)):
                    x_nt = jt.zeros([batch_size, self.emb_size])
                else:
                    x_nt = inputs_ta[t+1]
                t, x_t, s_t, finished = t+1, x_nt, s_nt, finished
            state = s_t
            outputs = jt.transpose(jt.stack(emit_ta), [1, 0, 2])
            return outputs, state

        def fgate_encoder(inputs, fields, inputs_len):
            batch_size = encoder_input.shape[0]
            hidden_size = self.hidden_size
            
            h0 = (jt.zeros([batch_size, hidden_size]),
                  jt.zeros([batch_size, hidden_size]))
            f0 = jt.zeros([batch_size], dtype=jt.int32)
            
            inputs_ta = jt.transpose(inputs, [1, 0, 2])
            fields_ta = jt.transpose(fields, [1, 0, 2])
            
            emit_ta=[]
            
            t, x_t, d_t, s_t, finished = 0, inputs_ta[0], fields_ta[0], h0, f0
            while(jt.reduce_add(finished) != f0.shape[0]):
                o_t, s_nt = self.enc_lstm(x_t, d_t, s_t, finished)
                emit_ta.append(o_t)
                finished = jt.greater_equal(t+1, inputs_len)
                if(jt.reduce_logical_and(finished)):
                    x_nt = jt.zeros([batch_size, self.uni_size])    
                    d_nt = jt.zeros([batch_size, self.field_attention_size])
                else:
                    x_nt=inputs_ta[t + 1]
                    d_nt=fields_ta[t + 1]       
                t, x_t, d_t, s_t,finished = t + 1, x_nt, d_nt, s_nt, finished
            state = s_t
            outputs = jt.transpose(jt.stack(emit_ta), [1, 0, 2])
            return outputs, state

        if self.fgate_enc:
            en_outputs, en_state = fgate_encoder(encoder_embed, field_pos_embed, encoder_len)
        else:
            en_outputs, en_state = encoder(encoder_embed, encoder_len)

        # ======================================== decoder ======================================== #
        
        def decoder_t(initial_state, inputs, inputs_len):
            batch_size = decoder_input.shape[0]

            h0 = initial_state
            f0 = jt.zeros([batch_size], dtype=jt.int32)
            x0 = self.embedding(jt.init.constant([batch_size], value=self.start_token))

            inputs_ta = jt.transpose(inputs, [1, 0, 2])
            emit_ta = []

            t, x_t, s_t, finished = 0, x0, h0, f0
            while(jt.reduce_add(finished) != f0.shape[0]):
                o_t, s_nt = self.dec_lstm(x_t, s_t, finished)
                if self.dual_att:
                    o_t, _ = self.att_layer(o_t, en_outputs, field_pos_embed)
                else:
                    o_t, _ = self.att_layer(o_t, en_outputs)
                o_t = self.dec_out(o_t, finished)
                emit_ta.append(o_t)
                finished = jt.greater_equal(t, inputs_len)
                if(jt.reduce_logical_and(finished)):
                    x_nt = jt.zeros([batch_size, self.emb_size])
                else:
                    x_nt = inputs_ta[t]
                t, x_t, s_t, finished = t+1, x_nt, s_nt, finished
            state = s_t
            outputs = jt.transpose(jt.stack(emit_ta), [1, 0, 2])
            return outputs, state

        def decoder_g(initial_state):
            batch_size = encoder_input.shape[0]

            h0 = initial_state
            f0 = jt.zeros([batch_size], dtype=jt.int32)
            x0 = self.embedding(jt.init.constant([batch_size], value=self.start_token))

            emit_ta = []
            att_ta = []

            t, x_t, s_t, finished = 0, x0, h0, f0
            while(jt.reduce_add(finished) != f0.shape[0]):
                o_t, s_nt = self.dec_lstm(x_t, s_t, finished)
                if self.dual_att:
                    o_t, w_t = self.att_layer(o_t, en_outputs, field_pos_embed)
                else:
                    o_t, w_t = self.att_layer(o_t, en_outputs)
                o_t = self.dec_out(o_t, finished)
                emit_ta.append(o_t)
                att_ta.append(w_t)
                next_token = jt.argmax(o_t, 1)[0]
                x_nt = self.embedding(next_token)
                finished = jt.logical_or(
                    finished, jt.equal(next_token, self.stop_token))
                finished = jt.logical_or(
                    finished, jt.greater_equal(t, self.max_length))
                t, x_t, s_t, finished = t+1, x_nt, s_nt, finished

            outputs = jt.transpose(jt.stack(emit_ta), [1, 0, 2])
            pred_tokens = jt.argmax(outputs, 2)[0]
            atts = jt.stack(att_ta)
            return pred_tokens, atts

        if is_train:
            # decoder for training
            de_outputs, de_state = decoder_t(en_state, decoder_embed, decoder_len)
            decoder_output = jt.reshape(decoder_output, (-1))
            de_outputs = jt.reshape(de_outputs, (-1, de_outputs.shape[2]))

            return jt.nn.cross_entropy_loss(de_outputs, decoder_output, ignore_index=0)
        else:
            # decoder for testing
            return decoder_g(en_state)
