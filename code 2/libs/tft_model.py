import tensorflow as tf
import numpy as np
import pandas as pd
import data_formatters.heart_rate
import libs.utils as utils

from tensorflow.keras.layers import Dense, Input, Add, Multiply, Dropout, Activation, LSTM

concat = tf.keras.backend.concatenate
stack = tf.keras.backend.stack
K = tf.keras.backend
LayerNorm = tf.keras.layers.LayerNormalization
InputTypes = data_formatters.heart_rate.InputTypes

def apply_gating_layer(x, hidden_layer_size, dropout_rate=None, activation=None):
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    activation_layer = Dense(hidden_layer_size, activation=activation)(x)
    gated_layer = Dense(hidden_layer_size, activation='sigmoid') (x)
    return Multiply() ([activation_layer, gated_layer])

def add_and_norm(x_list):
    tmp = Add() (x_list)
    return LayerNorm() (tmp)

def gated_residual_network(x, hidden_layer_size, output_size=None, dropout_rate=None, additional_context=None):
    if output_size is None:
        output_size = hidden_layer_size
        skip = x
    else:
        skip = Dense(output_size) (x)
    hidden = Dense(hidden_layer_size) (x)
    hidden = Activation('elu') (hidden)
    hidden = Dense(hidden_layer_size) (hidden)
    gating_layer = apply_gating_layer(hidden, output_size, dropout_rate=dropout_rate, activation=None)
    return add_and_norm([skip, gating_layer])

class TFTDataCache():
    _data_cache = {}
    
    @classmethod
    def update(cls, data, key):
        cls._data_cache[key] = data
    
    @classmethod
    def get(cls, key):
        return cls._data_cache[key].copy()
    
    @classmethod
    def contains(cls, key):
        return key in cls._data_cache        
        
class ScaledDotProductAttention():
    
    def __init__(self, attn_dropout=0.0):
        self.dropout = Dropout(attn_dropout)
        self.activation = Activation('softmax')
        
    def __call__(self, q, k, v):
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32')) # sqrt(d_k)
        attn = K.batch_dot(q, k, axes=[2,2]) / temper # [None, 30, 1] batch_dot [None, 30, 1] with axes [2,2] , result has shape [None, 30, 30]
        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = K.batch_dot(attn, v) # [None, 30, 30] dot [None, 30, 5] equals [None, 30, 5]
        
        return output # [None, 30, 5]

class InterpretableMultiHeadAttention():
    
    def __init__(self, n_head, d_model, dropout):
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model//n_head # key dimension = value dimension = d_model // n_heads
        self.dropout = dropout
        
        self.qs_layers, self.ks_layers, self.vs_layers = [], [], []
        vs_layer = Dense(d_v, use_bias=False)
        for _ in range(n_head):
            self.qs_layers.append(Dense(d_k, use_bias=False))
            self.ks_layers.append(Dense(d_k, use_bias=False))
            self.vs_layers.append(vs_layer)
        self.attention = ScaledDotProductAttention()
        self.w_o = Dense(d_model, use_bias=False)
        
    def __call__(self, q, k, v): # each [None, 30, 5]
        n_head = self.n_head
        heads = []
        for i in range(n_head):
            qs = self.qs_layers[i](q) # [None, 30, 5//n_head]
            ks = self.ks_layers[i](k) # [None, 30, 5//n_head]
            vs = self.vs_layers[i](v) # [None, 30, 5//n_head]
            head = self.attention(qs, ks, vs)
            head_dropout = Dropout(self.dropout) (head)
            heads.append(head_dropout)
        
        head = K.stack(heads) if n_head>1 else heads[0] # [n_head, None, 30, 5//n_head] or [None, 192, 5//n_head]
        outputs = K.mean(head, axis=0) if n_head>1 else head # [None, 30, 5//n_head]
        outputs = self.w_o(outputs) # [None, 192, 5]
        outputs = Dropout(self.dropout) (outputs)
        
        return outputs

class TemporalFusionTransformer():
    
    def __init__(self, raw_params):
        params = dict(raw_params)
        self.time_steps = int(params['total_time_steps'])
        self.input_size = int(params['input_size']) # number of input columns (excluding TIME and ID) -> 1+300=301
        self.output_size = int(params['output_size'])
        
        self._input_obs_loc = params['input_obs_loc']
        self._known_input_idx = params['known_inputs']        
        self.column_definition = params['column_definition']
        
        self.quantiles = [0.1, 0.5, 0.9]
        self.hidden_layer_size = int(params['hidden_layer_size'])
        self.dropout_rate = float(params['dropout_rate'])
        self.learning_rate = float(params['learning_rate'])
        self.minibatch_size = int(params['minibatch_size'])
        self.num_epochs = int(params['num_epochs'])
        self.early_stopping_patience = int(params['early_stopping_patience'])
        self.num_encoder_steps = int(params['num_encoder_steps'])
        self.num_stacks = int(params['stack_size']) # 1
        self.num_heads = int(params['num_heads']) # 4
        
        self.model = self.build_model()
    
    def get_tft_embeddings(self, all_inputs):
        def convert_real_to_embedding(x):
            return Dense(self.hidden_layer_size) (x)
        obs_inputs = K.stack( [ convert_real_to_embedding(all_inputs[Ellipsis, i:i+1]) for i in self._input_obs_loc], axis=-1 ) # stack: [None, 30, 5] -> [None, 30, 5, 1]
        known_inputs = K.stack( [ convert_real_to_embedding(all_inputs[Ellipsis, i:i+1]) for i in self._known_input_idx ], axis=-1 ) # stack: [None, 30, 5] -> [None, 30, 5, 300]        
        
        return known_inputs, obs_inputs # [None, 30, 5, 300], [None, 30, 5, 1]
    
    def _get_single_col_by_type(self, input_type):
        return utils.get_single_col_by_input_type(input_type, self.column_definition)
    
    def training_data_cached(self):
        return TFTDataCache.contains('train') and TFTDataCache.contains('valid')
    
    def _batch_data(self, data):
        
        def _batch_single_entity(input_data):
            time_steps = len(input_data)
            lags = self.time_steps # duration that we are going to trim down to
            x = input_data.values 
            if time_steps >= lags:
                return np.stack( [x[i:time_steps-(lags-1)+i] for i in range(lags)], axis=1 )
            else:
                return None
        
        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [tup[0] for tup in self.column_definition if tup[1] not in {InputTypes.ID, InputTypes.TIME}]
        
        data_map = {}
        for _, sliced in data.groupby(id_col):
            col_mappings = {'identifier': [id_col], 'time': [time_col], 'outputs': [target_col], 'inputs': input_cols}
            for k in col_mappings:
                cols = col_mappings[k]
                arr = _batch_single_entity(sliced[cols].copy())
                if arr is not None:
                    if k not in data_map:
                        data_map[k] = [arr]
                    else:
                        data_map[k].append(arr)
        for k in data_map:
            data_map[k] = np.concatenate(data_map[k], axis=0)
        data_map['full_outputs'] = data_map['outputs']
        data_map['outputs'] = data_map['outputs'][:, self.num_encoder_steps:, :]
        return data_map # dict of 4 entries 'identifier' [n_samples,30,1], 'inputs' [n_samples,30,301], 'outputs' [n_samples,10,1], 'time' [n_samples,30,1], each is 3D np array
    
    def cache_batched_data(self, data, cache_key):
        TFTDataCache.update(self._batch_data(data), cache_key)
    
    def _build_base_graph(self):
        time_steps = self.time_steps
        input_size = self.input_size
        encoder_steps = self.num_encoder_steps
        
        all_inputs = Input( shape=(time_steps, input_size) ) # [None, 30, 301]
        known_inputs, obs_inputs = self.get_tft_embeddings(all_inputs)
        historical_inputs = concat( [known_inputs[:, :encoder_steps, Ellipsis], obs_inputs[:, :encoder_steps, Ellipsis]], axis=-1 ) # [None, 20, 5, 301]
        future_inputs = known_inputs[:, encoder_steps:, :] # [None, 10, 5, 300]
        
        # apply variable selection network
        def lstm_combine_and_mask(embedding):
            _, time_steps, embedding_dim, num_inputs = embedding.get_shape().as_list() # 20, 5, 301
            flatten = K.reshape(embedding, [-1, time_steps, embedding_dim*num_inputs]) # [None, 20, 1505]
            mlp_outputs = gated_residual_network(flatten, self.hidden_layer_size, output_size=num_inputs, dropout_rate=self.dropout_rate) # [None, 20, 301]
            sparse_weights = Activation('softmax') (mlp_outputs)
            sparse_weights = tf.expand_dims(sparse_weights, axis=2) # [None, 20, 1, 301]
            
            trans_emb_list = []
            for i in range(num_inputs):
                grn_output = gated_residual_network(embedding[Ellipsis, i], self.hidden_layer_size, dropout_rate=self.dropout_rate) # [None, 20, 5]
                trans_emb_list.append(grn_output)
            transformed_embedding = stack(trans_emb_list, axis=-1) # [None, 20, 5, 301]
            combined = Multiply()([sparse_weights, transformed_embedding]) # [None, 20, 5, 301]
            temporal_ctx = K.sum(combined, axis=-1) # [None, 20, 5]
            return temporal_ctx
        
        historical_features = lstm_combine_and_mask(historical_inputs) # [None, 20, 5]
        future_features = lstm_combine_and_mask(future_inputs) # [None, 10, 5]
        
        def get_lstm(return_state):
            return LSTM(self.hidden_layer_size, return_state=return_state, return_sequences = True)
        
        history_lstm, state_h, state_c = get_lstm(return_state=True) (historical_features) # [None, 20, 5], [None, 5], [None, 5]
        future_lstm = get_lstm(return_state=False) (future_features, initial_state=[state_h, state_c]) # [None, 10, 5]
        lstm_layer = concat([history_lstm, future_lstm], axis=1) # [None, 30, 5]
        
        # apply gated skipped connection
        input_embeddings = concat([historical_features, future_features], axis=1) # [None, 30, 5]
        lstm_layer = apply_gating_layer(lstm_layer, self.hidden_layer_size, self.dropout_rate, activation=None) # [None, 30, 5]
        temporal_feature_layer = add_and_norm([lstm_layer, input_embeddings]) # [None, 30, 5]
        
        # we don't have static data, so we don't have static enrichment layer
        enriched = temporal_feature_layer
        # decode self_attention
        self_attn_layer = InterpretableMultiHeadAttention(self.num_heads, self.hidden_layer_size, dropout=self.dropout_rate)
        x = self_attn_layer(enriched, enriched, enriched) # [None, 30, 5]
        
        x = apply_gating_layer (x, self.hidden_layer_size, dropout_rate=self.dropout_rate) # [None, 30, 5]
        x = add_and_norm([x, enriched]) # [None, 30, 5]
        
        decoder = gated_residual_network(x, self.hidden_layer_size, dropout_rate=self.dropout_rate) # [None, 30, 5]
        
        decoder = apply_gating_layer(decoder, self.hidden_layer_size) # [None, 30, 5]
        transformer_layer = add_and_norm([decoder, temporal_feature_layer]) # [None, 30, 5]
        
        return transformer_layer, all_inputs # [None, 30, 5], [None, 30, 301]
        
            
    def build_model(self):
        time_steps = self.time_steps
        combined_input_size = self.input_size
        encoder_steps = self.num_encoder_steps
        
        transformer_layer, all_inputs = self._build_base_graph()
        outputs = Dense(self.output_size*len(self.quantiles)) (transformer_layer[Ellipsis, encoder_steps:, :]) # [None, 10, 3]
        adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model = tf.keras.Model(inputs=all_inputs, outputs=outputs)
        
        valid_quantiles = self.quantiles
        output_size = self.output_size
        
        class QuantileLossCalculator():
            
            def __init__(self, quantiles):
                self.quantiles = quantiles
                
            def quantile_loss(self, a, b):
                loss = 0.
                for i, quantile in enumerate(valid_quantiles):
                    loss += utils.tensorflow_quantile_loss(a[Ellipsis, output_size*i:output_size*i+1], b[Ellipsis, output_size*i:output_size*i+1], quantile )
                return loss
            
        quantile_loss = QuantileLossCalculator(valid_quantiles).quantile_loss
        model.compile(loss=quantile_loss, optimizer=adam, sample_weight_mode='temporal')
        self._input_placeholder = all_inputs
        return model
    
    def fit(self, train_df=None, data_formatter=None):
        train_data = TFTDataCache.get('train')
        
        def _unpack(data):
            tmp = data['outputs']
            labels = [ data_formatter._target_scaler.transform(tmp[i,:]) for i in range(tmp.shape[0]) ]
            return data['inputs'], np.stack(labels, axis=0)
        
        data, labels = _unpack(train_data)
        self.model.fit(x=data, y=np.concatenate([labels, labels, labels], axis=-1), epochs=self.num_epochs, batch_size=self.minibatch_size, shuffle=True, verbose=0)
        
    def predict(self):
        data = TFTDataCache.get('test')
        inputs = data['inputs']
        outputs = data['outputs']
        time = data['time']
        identifier = data['identifier']
        combined = self.model.predict(inputs) # [n_samples, output_duration, 3]; 3 because of 3 quantiles
        process_map = {'p{}'.format(int(q*100)) : combined[Ellipsis, i*self.output_size:(i+1)*self.output_size]
                       for i, q in enumerate(self.quantiles)} # each of 3 entries, each of size [42180, 24, 1]
        
        def format_outputs(prediction): # prediction has shape [n_samples, 10, 1]
            flat_prediction = pd.DataFrame( prediction[:,:,0], columns=[ 't+{}'.format(i) for i in range(self.time_steps-self.num_encoder_steps) ] )
            cols = list(flat_prediction.columns) # list of 24 columns 't+0', 't+1'
            flat_prediction['forecast_time'] = time[:, self.num_encoder_steps-1, 0] # time that we begin to predict
            flat_prediction['identifier'] = identifier[:,0,0]
            return flat_prediction[['forecast_time', 'identifier']+cols]
        
        process_map['targets'] = outputs
        
        ret = {k: format_outputs(process_map[k]) for k in process_map if k not in ['all_targets']} # k: p10, p50, p90, targets, each is a DataFrame [n_samples, 26]
        ret['all_targets'] = data['full_outputs']
        
        return ret
    
    def evaluate(self, eval_metrics='loss', data_formatter=None):
        test_data = TFTDataCache.get('test')
        
        def _unpack(data):
            tmp = data['outputs']
            labels = [ data_formatter._target_scaler.transform(tmp[i,:]) for i in range(tmp.shape[0]) ]
            return data['inputs'], np.stack(labels, axis=0)
        
        data, labels = _unpack(test_data)        
        metric_values = self.model.evaluate(x=data, y=np.concatenate([labels, labels, labels], axis=-1))
        metrics = pd.Series(metric_values, self.model.metrics_names) # self.model.metrics_names = ['loss']
        return metrics[eval_metrics]
    
    