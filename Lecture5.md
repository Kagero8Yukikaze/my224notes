# Lecture 5

## LSTM

- Long Short-Term Memory RNNs
  - 它的意思是long (short term)不是(long short) term，short term memory表示短时记忆，这里的long是相对于普通RNN而言的
- On step _t_, there is a **hidden state** $h^{(t)}$ and a **cell state** $c^{(t)}$
  - both are vectors length _n_
  - the cell stores long-term information (which is more similar to the hidden state in simple RNNs)
  - the LSTM can _read, erase, and write_ information from cell
- The selection of which information is erased/written/read is controlled by three corresponding **gates**
  - the gates are also vectors length _n_
  - on each timestep, each element of the gates can be open(1), closed(0), or somewhere in-between
  - the gates' value is computed base on the current context
- We have a sequence of inputs &x^{(t)}&, and we will compute a sequence of hidden states $h^{(t)}$ and cell state $c^{(t)}$. On timestep _t_:
$$
    \bold{f}^{(t)} = \sigma(\bold{W}_f \bold{h}^{(t-1)} + \bold{U}_f \bold{x}^{(t)} + \bold{b}_f)\\
    \bold{i}^{(t)} = \sigma(\bold{W}_i \bold{h}^{(t-1)} + \bold{U}_i \bold{x}^{(t)} + \bold{b}_i)\\
    \bold{o}^{(t)} = \sigma(\bold{W}_o \bold{h}^{(t-1)} + \bold{U}_o \bold{x}^{(t)} + \bold{b}_o)\\
    \tilde{\bold{c}}^{(t)} = tanh(\bold{W}_c \bold{h}^{(t-1)} + \bold{U}_c \bold{x}^{(t)} + \bold{b}_c)\\
    \bold{c}^{(t)} = \bold{f}^{(t)} \odot \bold{c}^{(t-1)} + \bold{i}^{(t)} \odot \tilde{\bold{c}}^{(t)}\\
    \bold{h}^{(t)} = \bold{o}^{(t)} \odot tanh \ \bold{c}^{(t)}
$$
  - $\bold{f}^{(t)}$ is **forget gate**: controls what is kept vs forgotten, from previous cell state (initialize to a 1 vector, which means preserve every thing)
  - $\bold{i}^{(t)}$ is **input gate**: controls what parts of the new cell content are written to cell (initialize to a 0 vector)
  - $\bold{o}^{(t)}$ is **output gate**: controls what parts of cell are output to hidden state
  - $\tilde{\bold{c}}^{(t)}$ is **new cell content**: this is the new content to be written to the cell
  - $\bold{c}^{(t)}$ is **cell state**: erase (“forget”) some content from last cell state, and write (“input”) some new cell content
  - $\bold{h}^{(t)}$ is **hidden state**: read (“output”) some content from the cell
  - $\sigma$ is sigmoid function
  - $\odot$ is element-wise(Hadamard) product
  - all these are vectors of **same length** _n_
- architecture:
![LSTM](pic/L5_pic1.jpg)
- solve vanishing gradients
  - LSTM makes it much easier for an RNN to preserve information over many steps
  - get about 100 timesteps rather than about 7 when using LSTM
  - lots of new deep feedforward/convolutional architectures add more direct connections:
    - ResNet
    - DenseNet
    - HighwayNet

## Bidirectional and Multi-layer RNNs

### Bidirectional RNN

- task: sentiment classification
![sentiment](pic/L5_pic2.jpg)
  - **contextual representation**: we can regard this hidden state as a representation of the word "terribly" in the context of this sentence
    - These contextual representations only contain information about the left context
  - so what about the right context?
![sentiment2](pic/L5_pic3.jpg)
  - on timestep _t_:
$$
    \overrightarrow{\bold{h}^{(t)}} = RNN_{BW}(\overrightarrow{\bold{h}^{(t)}}, \bold{x}^{(t)})\\
    \overleftarrow{\bold{h}^{(t)}} = RNN_{BW}(\overleftarrow{\bold{h}^{(t)}}, \bold{x}^{(t)})\\
    \bold{h}^{(t)} = [\overrightarrow{\bold{h}^{(t)}} ; \overleftarrow{\bold{h}^{(t)}}]
$$
    - $RNN_{FW}$ is a general notation to mean “compute one forward step of the RNN” – it could be a simple RNN or LSTM computation
    - generally, two RNNs have separate weights
  - simplified diagram:
![sentiment3](pic/L5_pic4.jpg)
    - The two-way arrows indicate bidirectionality and the depicted hidden states are assumed to be the concatenated forwards+backwards states
  - bidirectional RNNs are only applicable if you have access to **the entire input sequence** (so they are **not** applicable to Language Modeling)
  - Bidirectionality is powerful if you have entire input sequence!
