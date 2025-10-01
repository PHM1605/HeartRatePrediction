import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, Input, Add, Multiply, Dropout, Activation, LSTM, Lambda

concat = tf.keras.backend.concatenate
stack = tf.keras.backend.stack
K = tf.keras.backend
LayerNorm = tf.keras.layers.LayerNormalization

def add_and_norm(x_list):
    tmp = Add() (x_list)
    return LayerNorm() (tmp)

class FeedforwardNetwork():
    def __init__(self):
        self.inner_dimension = 512
        self.output_dimension = 5
    def __call__(self, x):
        self.inner = Dense(self.inner_dimension) (x) # [None, 20, 512]
        x = Activation('relu') (self.inner)
        self.outer = Dense(self.output_dimension) (x) # [None, 20, 5]
        return self.outer

class ScaledDotProductAttention():
    
    def __init__(self, attn_dropout=0.0):
        self.dropout = Dropout(attn_dropout)
        self.activation = Activation('softmax')
        
    def __call__(self, q, k, v):
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32')) # sqrt(d_k)
        attn = K.batch_dot(q, k, axes=[2,2]) / temper # q is x[0], k is x[1]; [None, len_decode, 1] batch_dot [None, len_encode, 1] = [None, len_decode, len_encode]
        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = K.batch_dot(attn, v) # [None,10,20] dot [None,20,1] equals [None,10,1]
        
        return output, attn # [None, 10, 1] and [None,20,20]
    
class MultiheadAttention():
    
    def __init__(self, n_head, d_model, dropout):
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model//n_head # dimension of key = dimension of value = dimension of model / number of heads
        self.dropout = dropout
        
        self.qs_layers, self.ks_layers, self.vs_layers = [], [], []
        vs_layer = Dense(d_v, use_bias=False) # use same Linear layer for "value" to support interpolation
        for _ in range(n_head):
            self.qs_layers.append(Dense(d_k, use_bias=False))
            self.ks_layers.append(Dense(d_k, use_bias=False))
            self.vs_layers.append(vs_layer)
            
        self.attention = ScaledDotProductAttention()
        self.w_o = Dense(d_model, use_bias=False)
        
    def __call__(self, q, k, v):
        n_head = self.n_head
        
        heads = []
        attns = []
        for i in range(n_head):
            qs = self.qs_layers[i](q) # [None,20,1]
            ks = self.ks_layers[i](k) # [None,20,1]
            vs = self.vs_layers[i](v) # [None,20,1]
            head, attn = self.attention(qs, ks, vs) # [None, 192, 5] and [None, 192, 192]
            head_dropout = Dropout(self.dropout)(head)
            heads.append(head_dropout)
            attns.append(attn)
        
        head = K.stack(heads) if n_head>1 else heads[0] # [n_head, None, 192, 5] or [None, 192, 5]
        attn = K.stack(attns) # [n_head, None, 192, 192]
        
        outputs = K.mean(head, axis=0) if n_head>1 else head # [None, 192, 5]
        outputs = self.w_o(outputs) # [None, 192, d_model]
        outputs = Dropout(self.dropout)(outputs)
        
        return outputs, attn

class transformer_encoder():
    def __init__(self):
        self.hidden_layer_size = 5
        self.n_layers = 2
        self.n_head = 4
        self.dropout = 0.1
        
        self.embed = Dense(self.hidden_layer_size)
        self.attention = MultiheadAttention(self.n_head, self.hidden_layer_size, self.dropout)
        self.feedforward = FeedforwardNetwork()
    def __call__(self, x):
        embed = self.embed (x) # [None, 20, 5]
        for _ in range(self.n_layers):            
            attn_out, _ = self.attention(embed, embed, embed) # [None, 20, 5]
            forward_input = add_and_norm([attn_out, embed]) # [None, 20, 5]
            forward_output = self.feedforward(forward_input)
            embed = add_and_norm([forward_input, forward_output])
        return embed
    
class transformer_decoder():
    def __init__(self):
        self.hidden_layer_size = 5
        self.n_layers = 2
        self.n_head = 4
        self.dropout = 0.1
        
        self.embed = Dense(self.hidden_layer_size)
        self.attention_1 = MultiheadAttention(self.n_head, self.hidden_layer_size, self.dropout)
        self.attention_2 = MultiheadAttention(self.n_head, self.hidden_layer_size, self.dropout)
        self.feedforward = FeedforwardNetwork()
        self.dec_o = Dense(self.hidden_layer_size)
        self.w_o = Dense(1, use_bias=False)
    def __call__(self, x, encoder_output, padding=True):
        if padding==True:
            x = tf.pad(x, [[0,0],[1,0],[0,0]] ) # padding one '0' before
        embed = self.embed (x) # [None,11,5]
        for _ in range(self.n_layers):            
            attn_outp, _ = self.attention_1(embed, embed, embed) # [None,11,5]
            attn_outp = add_and_norm([attn_outp, embed]) # [None,11,5]
            attn_outp_2, _ = self.attention_2(attn_outp, encoder_output, encoder_output)
            forward_input = add_and_norm([attn_outp_2, attn_outp])            
            forward_output = self.feedforward(forward_input)
            embed = add_and_norm([forward_input, forward_output])
        return embed # [None,11,5]

class transformer_linear_out():
    def __init__(self):
        self.hidden_layer_size = 5
        self.dec_o = Dense(self.hidden_layer_size)
        self.w_o = Dense(1, use_bias=False)
    def __call__(self, x):
        tmp = self.dec_o(x)
        return self.w_o(tmp)

class Transformer():
    
    def __init__(self):
        self.time_steps = 30
        self.input_size = 11
        self.output_size = 1
        
        self.quantiles = [0.1, 0.5, 0.9]
        self.hidden_layer_size = 5
        self.dropout_rate = 0.1
        self.learning_rate = 0.001
        self.minibatch_size = 32
        self.num_epochs = 1000
        self.num_encoder_steps = 20
        
        self.encoder = transformer_encoder()
        self.decoder = transformer_decoder()
        self.linear_out = transformer_linear_out()
        self.model = self.build_model()
 
    def build_model(self):
        
        time_steps = self.time_steps
        encoder_steps = self.num_encoder_steps
        self.decoder_steps = time_steps-encoder_steps
        combined_input_size = self.input_size

        # model for training
        encoder_input = Input((encoder_steps, combined_input_size)) # [None, 20, 11]
        decoder_input = Input((self.decoder_steps, 1)) # [None, 10, 1]
        
        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output) # [None, 11, 1]
        
        # output linear layers
        linear_o = self.linear_out(decoder_output)
        training_model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=linear_o)
        training_model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        
        return training_model
    
    def fit(self, data):
        data_input = data['inputs'][:, :self.num_encoder_steps, :]
        data_target = data['outputs'][:, :, 0]
        self.model.fit(x=[data_input, data_target], y=np.pad(data_target, pad_width=[(0,0),(0,1)]), epochs=self.num_epochs, batch_size=self.minibatch_size, shuffle=True, verbose=0)
        
    def predict(self, data):
        inputs = data['inputs'][:, :self.num_encoder_steps, :]
        outputs = data['outputs']   
        
        def predict_sequence(input_seq):
            output_seq = []
            encoder_outp = self.encoder(input_seq)   
            target_seq = np.zeros((1,1,1))
            for i in range(self.decoder_steps):                 
                 decoder_outp = self.decoder(target_seq, encoder_outp, padding=False) # [1,1,5]
                 outp = self.linear_out(decoder_outp)
                 output_seq.append(outp[0,0,0].numpy())
                 
                 target_seq[0,0,0] = outp[0,0,0].numpy()
                 #encoder_outp = decoder_outp
            return np.array(output_seq)
        
        pred_ret = []
        for i in range(inputs.shape[0]):
            tmp = predict_sequence(inputs[i:i+1, Ellipsis])
            pred_ret.append(tmp)
        pred_ret = np.stack(pred_ret)
        
        process_map = {}
        process_map['predictions'] = pred_ret
        process_map['targets'] = outputs[:,:,0]
        process_map['all_targets'] = data['full_outputs'][:,:,0]
        return process_map
    
    def evaluate(self, y_true, y_pred):
        
        def numpy_quantile_loss(y, y_pred, quantile):
            prediction_underflow = y - y_pred
            q_loss = quantile * np.maximum(prediction_underflow, 0) + (1.-quantile) * np.maximum(-prediction_underflow, 0.)
            q_loss_arr = np.sum(q_loss, axis=-1)
            return np.mean(q_loss_arr)
        
        quantiles = self.quantiles
        loss = []
        for quantile in quantiles:
            loss.append( numpy_quantile_loss( y_true, y_pred, quantile) )
        return loss