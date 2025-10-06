import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Lambda

from tensorflow.keras.layers import Dense, Input, Add, Multiply, Dropout, Activation, LSTM

concat = tf.keras.backend.concatenate
stack = tf.keras.backend.stack
K = tf.keras.backend
LayerNorm = tf.keras.layers.LayerNormalization

class Sequence2Sequence():
    
    def __init__(self, quantile):
        self.time_steps = 30
        self.input_size = 11
        self.output_size = 1
        
        self.quantile = quantile
        self.hidden_layer_size = 5
        self.dropout_rate = 0.1
        self.learning_rate = 0.001
        self.minibatch_size = 32
        self.num_epochs = 1000
        self.num_encoder_steps = 20
        
        self.model, self.encoder, self.decoder = self.build_model()
 
    def build_model(self):
        
        time_steps = self.time_steps
        encoder_steps = self.num_encoder_steps
        self.decoder_steps = time_steps-encoder_steps
        combined_input_size = self.input_size

        # model for training
        encoder_input = Input((encoder_steps, combined_input_size)) # [None, 20, 11]
        decoder_input = Input((self.decoder_steps, 1)) # [None, 10, 1]
        
        paddings = tf.constant([[0, 0], [1, 0], [0, 0]]) # pad 1 column BEFORE the 2nd dimension
        # decoder_input_padded = tf.pad( decoder_input, paddings, 'CONSTANT') # [None, 11, 1]
        decoder_input_padded = Lambda(lambda x: tf.pad(x, [[0,0],[1,0],[0,0]], "CONSTANT"))(decoder_input)
        
        input_embedding = Dense(self.hidden_layer_size)
        encoder_input_embed = input_embedding (encoder_input) # [None, 20, 5]
        encoder = LSTM(self.hidden_layer_size, return_state=True)
        encoder_output, h, c = encoder(encoder_input_embed)
        encoder_states = [h, c]
        
        output_embedding = Dense(self.hidden_layer_size)
        decoder_input_embed = output_embedding (decoder_input_padded) # [None, 11, 5]
        decoder = LSTM(self.hidden_layer_size, return_sequences=True, return_state=True)
        decoder_output, _, _ = decoder(decoder_input_embed, initial_state=encoder_states) # [None, 11, 5]
        decoder_dense = Dense(1)
        decoder_output = decoder_dense(decoder_output) # [None, 11, 1]
   
        def quantile_loss(y_true, y_pred):
            prediction_underflow = y_true - y_pred
            q_loss = self.quantile * tf.maximum(prediction_underflow, 0) + (1.-self.quantile) * tf.maximum(-prediction_underflow, 0.)
            q_loss_arr = tf.reduce_sum(q_loss, axis=-1)
            return tf.reduce_mean(q_loss_arr)
                    
        training_model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)
        training_model.compile(loss=quantile_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        
        # model for predicting
        prediction_encoder = tf.keras.Model(encoder_input, encoder_states)
        decoder_state_input_h = Input(shape=(self.hidden_layer_size,))
        decoder_state_input_c = Input(shape=(self.hidden_layer_size,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_input_single = Input(shape=(1,))
        expanded_single = Lambda(lambda x: tf.expand_dims(x,axis=-1))(decoder_input_single)
        decoder_input_single_embedded = output_embedding(expanded_single)
        # decoder_input_single_embedded = output_embedding(tf.expand_dims(decoder_input_single, axis=-1))
        decoder_outputs, h, c = decoder(decoder_input_single_embedded, initial_state=decoder_states_inputs)
        decoder_states = [h,c]        
        prediction  = decoder_dense(decoder_outputs)
        prediction_decoder = tf.keras.Model(inputs=[decoder_input_single]+decoder_states_inputs, outputs=[prediction]+decoder_states)
        
        return training_model, prediction_encoder, prediction_decoder
    
        # paddings = tf.constant([[0, 0], [0, 1]) # pad 1 column AFTER the 2nd dimension
        #decoder_output_padded = tf.pad( decoder_input, paddings, 'CONSTANT') # [None, 11, 1]     
    
    
    def fit(self, data):
        data_input = data['inputs'][:, :self.num_encoder_steps, :]
        data_target = data['outputs'][:, :, 0]
        self.model.fit(x=[data_input, data_target], y=np.pad(data_target, pad_width=[(0,0),(0,1)]), epochs=self.num_epochs, batch_size=self.minibatch_size, shuffle=True, verbose=0)
        
    def predict(self, data):
        inputs = data['inputs'][:, :self.num_encoder_steps, :]
        outputs = data['outputs']   
        
        def predict_sequence(input_seq):
            output_seq = []
            state_values = self.encoder.predict(input_seq)   
            target_seq = np.zeros((1,1))
            for i in range(self.decoder_steps):                 
                 outp, h, c = self.decoder.predict([target_seq] + state_values)
                 output_seq.append(outp[0,0,0])
                 target_seq[0,0] = outp[0,0,0]
                 state_values = [h,c]
            return np.array(output_seq)
        
        pred_ret = []
        for i in range(inputs.shape[0]):
            tmp = predict_sequence(inputs[i:i+1,:])
            pred_ret.append(tmp)
        pred_ret = np.stack(pred_ret)
        
        process_map = {}
        process_map['predictions'] = pred_ret
        process_map['targets'] = outputs[:,:,0]
        process_map['all_targets'] = data['full_outputs'][:,:,0]
        return process_map
    
    def evaluate(self, y_true, y_pred):
        
        def numpy_quantile_loss(y, y_pred):
            prediction_underflow = y - y_pred
            q_loss = self.quantile * np.maximum(prediction_underflow, 0) + (1.-self.quantile) * np.maximum(-prediction_underflow, 0.)
            q_loss_arr = np.sum(q_loss, axis=-1)
            return np.mean(q_loss_arr)
        
        loss = numpy_quantile_loss( y_true, y_pred)
        return loss