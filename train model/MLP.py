from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from dataHelper import datareader
import matplotlib.pyplot as plt

data = datareader.Data('','','','','','','','')
data.data_process()


model = Sequential()
model.add(Embedding(output_dim=32,
                    input_dim=2000,
                    input_length=100))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=256,
                activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,
                activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


train_history = model.fit(data.train,data.train_label,batch_size=100,
                          epochs=10,verbose=2,
                            validation_split=0.2
                          )
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='uppereft')
    plt.show()
show_train_history(train_history,'accuracy','val_accuracy')
show_train_history(train_history,'loss','val_loss')