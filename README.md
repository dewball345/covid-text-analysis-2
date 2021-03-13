# covid-text-analysis-2
Model files for covid tweet sentiment analysis


# How to use
First, make sure to import ```@tensorflow/tfjs```
```
npm init
npm install @tensorflow/tfjs
```
## Key Functions

```
import * as tf from '@tensorflow/tfjs';

function convert_to_tensors(word, word_index){
  word = word.split(" ")
  word = word.map(element => {
    if(word_index[element] !== undefined){
      return word_index[element]
    } else {
      return word_index[1]
    }
  });
  let n = 64-word.length;
  let padding = new Array(n); for (let i=0; i<n; ++i) padding[i] = 0;
  word = padding.concat(word)
  word = tf.tensor2d([word], [1, word.length])
  return word;
}

function indexOfMax(arr) {
  if(arr.length === 0){
    return -1
  }
  let arrindex = 0
  for(let index = 0; index < arr.length; index++ ){
    if(arr[index] > arr[arrindex]){
      arrindex = index
    } 
  }

  return arrindex
}

```

## Include this code to make predictions

```
async function main(tweet){
  const model = await tf.loadLayersModel("https://raw.githubusercontent.com/dewball345/covid_text_sentiment_analysis/main/model.json");
  console.log(model.summary())
  let word_index = await fetch('https://raw.githubusercontent.com/dewball345/covid-text-analysis-2/main/word_index.json')
  word_index = await word_index.json();
  let input = convert_to_tensors(tweet, word_index);
  let classes = ["Sad", "Happy", "Little Sad", "Neutral", "Little Happy"]
  let prediction = Array(this.state.model.predict(input).dataSync())[0];
  let maxIndex = indexOfMax(prediction)
  console.log(classes[maxIndex])
}

//Should output "Little Sad"
main("i really hate covid-19");
```


