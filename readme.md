# CultureLLM: Fine-tuning Culture-aware Large Language Models with Semantic Data Augmentation

Large language models (LLMs) are reported to be partial to certain cultures and thus suffer from culture difference. While cultural data are often expensive to collect,  existing efforts handles this challenge by prompt engineering or culture-specific pre-training. However, they might overlook the culture specificity and require extensive computing resources. In this paper, we propose CultureLLM, a cost-effective solution to fine-tune culture-aware LLMs. CultureLLM adopts World Value Survey as the seed data and then generates diverse and semantically equivalent training data using the proposed semantic data augmentation approach. Using only $50$ seed samples from WVS with augmented data, we fine-tuned culture-specific LLMs and one unified model (CultureLLM-One) for 9 cultures covering rich and low-resource languages. Extensive experiments on 59 culture-related datasets demonstrate that CultureLLM significantly outperforms various counterparts such as ChatGPT and Gemini Pro with comparable or even better performance than GPT-4. Our human study shows that the generated samples are diverse and semantically equivalent to the original samples, providing an effective solution for LLMs augmentation.

## Install

```bash
pip install jsonlines fire scikit-learn torch==2.0.0 transformers bitsandbytes accelerate
pip install openai
pip instal nltk
pip install -q -U google-generativeai
```
## Data Augmentation

```bash
python main.py --n 5 --m 10
```

Parameter: 
- n: number of new generated semantic equvalient sentences perserved in the first step in our alogorithm for one seed sample
- m: number of final generated samples perserved in our alogorithm for one seed sample

## Dataset for Fine-tuning and Experiments

- /data contains all datasets for fine-tuning and experiments.
- /data/WVQ.jsonl contains seed data from World Values Survey.
- /data/new_WVQ_500.jsonl contains 500 new generated samples via our data augmentation approach. Those data samples are also be used in our main experiments.
- /data/new_WVQ_100.jsonl contains 100 new generated samples via our data augmentation approach.
- /data/new_WVQ_1000.jsonl contains 1000 new generated samples via our data augmentation approach.
- /data/new_WVQ_sentence_only.jsonl contains 500 new generated samples via the first steps in our data augmentation approach.

Besides, there are nine directories in /data, containing datasets both for fine-tuning and experiments for specific culture. /Finetune contains different versions of training datasets, which are distinguished by the last characters in file names. 
- "50", "150", "no suffix", and "1000" represents fine-tuning via 50, 150, 500 and 1000 new generated samples, respectively. 
- "L" respresents fine-tuning via 500 new generated samples in specific language. 
- "llama" represents fine-tuning data for Llama. 
- "sentence_only" represents fine-tuning via 500 new generated samples by the first steps in our data augmentation approach.

## Fine-tuning your own CultureLLM

data_process.py is to process data for fine-tuning. 

### Fine-tune the same CultureLLMs as mentioned in paper

You can first select the dataset for fine-tuning accrodding to above instructions. Then run finetune() function in data_process.py.

### Fine-tune other CultureLLMs

- Step 1: download WVS file from https://www.worldvaluessurvey.org/WVSContents.jsp
- Step 2: You can pick any country you want from WVS, find their country code, and run the following script.
```bash
python data_process.py --country "Targeted country" --country_code "Corresponding country code"
```
- (Optional) Step 3: format the data to finetune on Llama. Run generateData4Llama() function.
- (Optional) Step 4: Translate the data into other language. Run processLanguage() function.
- Step 5: Fine-tuning.

## Fine-tune CultureLLM-Llama-70b-chat

```bash
python llama_finetune.py --base_model "path_of_llama_70b" --new_model "path_of_new_model" --data_files "fine-tuning data path"
```

## Experiments

You can reproduce all experiments in our paper.
- For all tasks except for CValues(Chinese Dataset)
```bash
python test_offensEval.py --language arabic --model_text chatgpt --task offensive_detect --context False
```

Parameter: 
- language: the language of targeted culture. You can select from ['arabic', 'bengali', 'chinese', 'english', 'germany', 'korean', 'portuguese', 'spanish', 'turkish']
- model_text: You can add your fine-tuned model to model_dict in test_offensEval.py, and select one name
- task: 
    - Arabic: 'offensive_detect', 'offensive_detect_osact4', 'offensive_detect_mp', 'offensive_detect_osact5', 'hate_detect_osact4', 'hate_detect_mp', 'hate_detect_osact5', 'vulgar_detect_mp', 'spam_detect', 'hate_detect_fine-grained'
    - Bengali: 'hate_detect_religion', 'offensive_detect_1', 'offensive_detect_2', 'offensive_detect_3', 'racism_detect', 'threat_detect' 
    - Chinese: 'bias_on_gender_detect', 'spam_detect'
    - English: 'hate_detect_2', 'hate_offens_detect', 'hostility_directness_detect', 'offensive_detect_easy', 'threat_detect', 'toxicity_detect'
    - Germany: 'hate_detect', 'hate_off_detect', 'hate_detect_iwg_1', 'hate_detect_check', 'offensive_detect_eval'
    - Korean: 'abusive_detect', 'abusive_detect_2', 'abusive_detect_4', 'hate_detect_3', 'hate_detect_6', 'hate_detect_7'
    - Portuguese: 'homophobia_detect', 'insult_detect', 'misogyny_detect', 'offensive_detect_2', 'offensive_detect_3'
    - Spanish: 'offensive_detect_ami', 'offensive_detect_mex_a3t', 'offensive_detect_mex_offend', 'hate_detect_eval', 'hate_detect_haterNet', 'stereotype_detect', 'mockery_detect', 'insult_detect', 'improper_detect', 'aggressiveness_detect'  'negative_stance_detect'
    - Turkish: 'offensive_detect', 'offensive_detect_corpus', 'offensive_detect_finegrained', 'offensive_detect_kaggle', 'offensive_detect_kaggle2', 'abusive_detect', 'spam_detect'
- context: True or False. If True, a culture-related context will be appended in prompt.
- For CValues(Chinese Dataset)
```bash
python test_CValues.py --model_text chatgpt --context False
```

Parameter: 
- model_text: You can add your fine-tuned model to model_dict in test_CValues.py, and select one name.
- context: True or False. If True, a culture-related context will be appended in prompt.