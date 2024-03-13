# Natural Language Processing
Some models I have implemented from scratch in PyTorch using various tutorials and papers. 

## List of models
+ [x] Seq2Seq
+ [ ] Seq2Seq with attention (work in progress)
+ [ ] Transformer architecture (based on the `Attention is all you need' paper [1])
+ [ ] GPT-like model from scratch

# Model performance
## Seq2Seq
Training params:
+ epochs: 20
+ learning rate: 1e-3
+ batch size: 64
+ input language: german (de)
+ output language: english (en)
+ shuffle batches: true
+ encoder embedding size: 300
+ decoder embedding size: 300
+ hidden size: 1024
+ num layers: 2
+ encoder dropout: 0.5
+ decoder dropout: 0.5
+ teacher forcing percentage: 0.5

Some examples (I came up with) after 20 epochs using the Multi30K dataset:
| Input                                                                            | Model output                                                   | Correct translation                                                     |
|----------------------------------------------------------------------------------|----------------------------------------------------------------|-------------------------------------------------------------------------|
| Ein Mann läuft entlang eine Straße.                                              | A man walking down a street.                                   | A man walking down a street.                                            |
| Eine Frau läuft entlang eine Straße.                                             | A woman walking down a street.                                 | A woman walking down a street.                                          |
| Er ist groß.                                                                     | A is a man.                                                    | He is large.                                                            |
| Ein Mann und zwei pferden ziehen einen Auto aus dem Meer.                        | A man and two men are a out of a large body of water           | A man and two horses are pulling a car out of the sea.                  |
| Ein Skateboarder macht einen coolen Trick.                                       | A skateboarder makes a trick trick.                            | A skateboader does a cool trick.                                        |
| Hunderte Kinder besturmen einen Mann.                                            | A group of people looking at a man.                            | Hundreds of children storming a man.                                    |
| Ein sehr langer Satz der das Netzwerk wahrscheinlich nicht gut verarbeiten kann. | A very very very colorful the the price Angels for the basket. | A very long sentence which the network probably can't handle very well. |

The network is actually performing quite well for such a simple architecture! Longer sentences are harder for the network. I also think the dataset is biased towards sentences which are descriptions of things (like "A man walking down a street" or "two dogs taking a bath") so inputs not of this form behave a lot worse (like "Er ist groß").

Nontheless a nice result.
## References
[1] - Attention is all you need - A. Vaswani et al. - (12 Jun 2017) - DOI: https://doi.org/10.48550/arXiv.1706.03762
