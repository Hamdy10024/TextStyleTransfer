import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from attention import attention
import data_helpers
import numpy as np
import params

class SiameseDiscriminator(object):
   
    def __init__(self, embedding_size, init_embed, hidden_size, \
                 attention_size, max_sent_len, keep_prob, just_embed = True):
        # training inputs
        self.input_x = tf.placeholder(tf.int32, [None, None, max_sent_len], name="input_x")
        self.sequence_length = tf.placeholder(tf.int32, [None,None], name="input_len")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
         
        self.input_em = tf.placeholder(tf.int32, [None, max_sent_len], name="input_x_em")
        self.sequence_len_em =  tf.placeholder(tf.int32, [None], name="input_len_em")
        self.is_train = tf.placeholder(tf.int32,(),name="is_train")
        with tf.variable_scope('siamese_discriminator'):
            # embedding layer with initialization
            batch_size = tf.shape(self.input_x)[1]
            num_classes = tf.shape(self.input_x)[0]
            with tf.name_scope("pair_inps"):
                    self.input ,self.sequence_len , self.labels= self.all_class_flattener(self.input_x,self.sequence_length,self.is_train)
            with tf.name_scope("flatten_input"):
                self.inter_inp = self.merge_sents(self.input)
                self.inner_lens = tf.reshape(self.sequence_len, [num_classes*batch_size* 4])
            with tf.name_scope("embedding"):
                # trainable embedding
                W = tf.Variable(init_embed, name="W", dtype=tf.float32)
                self.embedded_chars = tf.nn.embedding_lookup(W, self.inter_inp)
                self.embedded_chars_em = tf.nn.embedding_lookup(W, self.input_em)
            # RNN layer + attention
            with tf.name_scope("bi-rnn"):
                self.gru1 = GRUCell(hidden_size)
                self.gru2 = GRUCell(hidden_size)
                rnn_outputs, _ = bi_rnn(self.gru1, self.gru2 ,\
                                        inputs=self.embedded_chars, sequence_length=self.inner_lens, \
                                        dtype=tf.float32)
                rnn_outputs_em, _ = bi_rnn(self.gru1, self.gru2 ,\
                                        inputs=self.embedded_chars_em, sequence_length=self.sequence_len_em , \
                                        dtype=tf.float32)
              
                self.attention_outputs, self.alphas = attention(rnn_outputs, attention_size, return_alphas=True)
                
                self.attention_outputs_em, self.alphas_em = attention(rnn_outputs_em, attention_size, return_alphas=True)
                self.output_em = tf.reduce_mean(self.attention_outputs_em, axis = 0)
                drop_outputs = tf.nn.dropout(self.attention_outputs, keep_prob)
            with tf.name_scope('flattener'):
                self.drop_outputs = tf.reshape(drop_outputs, (num_classes * batch_size*2,2,-1)) #b,2,d
            
            with tf.name_scope('similarity_measure'):
                #
                self.d1 =d1 =   self.distance(self.drop_outputs[:,0], self.drop_outputs[:,1])
                loss = self.labels * tf.square(d1) +(1- self.labels)* tf.square(tf.maximum((1 - d1),0))
                self.loss =  tf.div(tf.reduce_mean(loss),2)
            with tf.name_scope("accuracy"):
                self.temp_sim = tf.subtract(tf.ones_like(self.d1),tf.rint(self.d1), name="temp_sim") #auto threshold 0.5
                correct_predictions = tf.equal(self.temp_sim, self.labels)
                self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
           

        self.params = [param for param in tf.trainable_variables() if 'siamese_discriminator' in param.name]
        for param in self.params:
            print(param.name)
        sd_optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = sd_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = sd_optimizer.apply_gradients(grads_and_vars)
            
    def split_sents(self, inputs, num_classes):
        return tf.reshape(inputs, (num_classes,tf.shape(inputs)[0] / num_classes,-1))

    def merge_sents(self, inputs):
        return tf.reshape(inputs, (tf.shape(inputs)[0] * tf.shape(inputs)[1], -1))
    def distance(self,a, b):
      dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(a, b)),1,keep_dims=True))
      dist = tf.div(dist, tf.add(tf.sqrt(tf.reduce_sum(tf.square(a),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(b),1,keep_dims=True))))
      dist = tf.reshape(dist, [-1], name="distance")
      return dist
  
    def getSiameseReward(self, sess, sents, sents_len, target_sents, tsf_lens):
        inp,lenses = data_helpers.generatePretrainSiameseSamples(sents,target_sents,sents_len,tsf_lens,0,min(len(sents),len(target_sents)))
     
        feed = {self.input_x: inp, self.sequence_length: lenses, self.is_train: 0}
        rewards = sess.run(self.d1, feed_dict=feed)
        rewards = np.reshape(rewards, (-1,))
        correct_range = np.arange(len(sents)) * 2
        rewards =  np.array(rewards)[correct_range]
        return 1 - rewards
    
    def feat_extract(self,sess,inputs,lengths):
        feed = {self.input_em: inputs,self.sequence_len_em: lengths}
        embedd = sess.run(self.output_em, feed_dict = feed)
        return embedd
        
        
    
    def siamese_flattener(self, inputs, lens,class_num): #num_classes, batch_size, max_len
      num_classes = tf.shape(inputs)[0]
      batch_size = tf.shape(inputs)[1]
      max_len  = tf.shape(inputs)[2]
      idx = tf.constant(0)
      outputs = tf.zeros([1,2,max_len],tf.int32)
      labels = tf.zeros([1],tf.float32)
      lenses = tf.zeros([1,2],tf.int32)
      def diff_sent(class_num):
        rand_class = tf.random.uniform([1], 0,num_classes - 1, tf.int32)[0]
        rand_class = (rand_class + class_num + 1) %num_classes
        rand_index = tf.random.uniform([1], 0,batch_size, tf.int32)[0]
        return tf.gather_nd(inputs,(rand_class,rand_index)), lens[rand_class,rand_index]
      def match_inst(idx, outputs, lenses,class_num, labels):
        orig_sent = tf.expand_dims(tf.gather_nd(inputs,(class_num,idx)),0)
        orig_len = lens[class_num,idx]
        rand_index = tf.random.uniform([1], 0,batch_size, tf.int32)
        sim_sent = tf.expand_dims(tf.gather_nd(inputs,(class_num ,rand_index[0])), 0)
        sim_sent_len = lens[class_num ,rand_index[0]]
        diff_s, diff_l = diff_sent(class_num)
        diff = tf.expand_dims(diff_s,0)
        inst_arr = tf.concat([orig_sent, sim_sent], 0)
        inst_arr2 = tf.concat([orig_sent, diff], 0)
        len_arr = tf.stack([orig_len,sim_sent_len])
        len_arr2 = tf.stack([orig_len, diff_l])
        outputs = tf.concat([outputs, tf.expand_dims(inst_arr,0)],0)
        outputs = tf.concat([outputs, tf.expand_dims(inst_arr2,0)],0)
        lenses = tf.concat([lenses, tf.expand_dims(len_arr,0)],0)
        lenses = tf.concat([lenses, tf.expand_dims(len_arr2,0)],0)
        labels = tf.concat([labels, [1,0]],0)
        return [idx +1 , outputs, lenses,class_num, labels]
      condition_func = lambda idx, outputs,lenses, class_num, labels: idx < batch_size
      _, outputs,lenses,_,labels = tf.while_loop(
            condition_func, match_inst,
            loop_vars=[idx, outputs,lenses, class_num, labels],
            shape_invariants=[idx.get_shape(), tf.TensorShape(None),tf.TensorShape(None),class_num.get_shape(), tf.TensorShape(None)])
      outputs = outputs[1:]
      lenses = lenses[1:]
      labels = labels[1:]
      return outputs,lenses,labels #batch_size,X2X,max_len
        
    def all_class_flattener(self,inputs,lengths,is_train=tf.constant(1)):
      num_classes = tf.shape(inputs)[0]
      max_len  = tf.shape(inputs)[2]
      outputs = tf.zeros([1,2,max_len],tf.int32)
      labels = tf.zeros([1],tf.float32)
      lenses = tf.zeros([1,2],tf.int32)
      class_num = tf.constant(0)
      def body_func(class_num, outputs,lenses , labels):
        outs,lens ,label= self.siamese_flattener(inputs,lengths,class_num)
        return [class_num + 1, tf.concat([outputs, outs],0), tf.concat([lenses, lens],0), tf.concat([labels,label],0)]
      condition_func = lambda idx, outputs,lenses, labels: idx < num_classes
      _, outputs,lenses,labels = tf.while_loop(
            condition_func, body_func,
            loop_vars=[class_num, outputs,lenses,labels],
            shape_invariants=[class_num.get_shape(), tf.TensorShape(None),tf.TensorShape(None),tf.TensorShape(None)])
      outputs = outputs[1:]
      lenses = lenses[1:]
      labels = labels[1:]
      if is_train == 0:
          shuffled_indices = tf.random.shuffle(tf.range(tf.shape(outputs)[0],dtype = tf.int32),name="shuffler")
          outputs = tf.gather(outputs, shuffled_indices)
          lenses = tf.gather(lenses, shuffled_indices)
          labels = tf.gather(labels, shuffled_indices)
      return outputs,lenses,labels #batch_size*num_classes,X2X->3,max_len

   







