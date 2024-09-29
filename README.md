# HelpMum-Personal/ai-translator

This model is a fine-tuned version of Facebook's [M2M100](https://huggingface.co/facebook/m2m100_418M) translation model, optimized for translating from Yoruba, Igbo, and Hausa into English. It was trained on a dataset consisting of 1,500,000 sentences (500,000 for each language), providing high-quality translations for these languages.
It was built with the intention of building a system that makes it easier to communicate with LLMs using Igbo, Hausa and Yoruba languages.

## Model Details

- **Base Model**: [Facebook M2M100](https://huggingface.co/facebook/m2m100_418M)
- **Languages Supported**: 
  - Source Language: Yoruba, Igbo, Hausa 
  - Target Languages: English



### Model Usage

To use this model for translation tasks, you can load it from Hugging Face’s `transformers` library:

```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load the fine-tuned model
model = M2M100ForConditionalGeneration.from_pretrained("HelpMum-Personal/m2m100_418M-nig-en")
tokenizer = M2M100Tokenizer.from_pretrained("HelpMum-Personal/m2m100_418M-nig-en")

# translate igbo to English
igbo_text="Nlekọta ahụike bụ mpaghara dị mkpa n'ihe fọrọ nke nta ka ọ bụrụ obodo ọ bụla n'ihi na ọ na-emetụta ọdịmma na ịdịmma ndụ nke ndị mmadụ n'otu n'otu. Ọ gụnyere ọtụtụ ọrụ na ọrụ dị iche iche, gụnyere nlekọta mgbochi, nchoputa, ọgwụgwọ na njikwa ọrịa na ọnọdụ. Usoro nlekọta ahụike dị mma na-achọ imeziwanye nsonaazụ ahụike, belata ọrịa ọrịa, yana hụ na ndị mmadụ n'otu n'otu nwere ohere ịnweta ọrụ ahụike dị mkpa."
tokenizer.src_lang = "ig"
tokenizer.tgt_lang = "en"
encoded_ig = tokenizer(igbo_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_ig, forced_bos_token_id=tokenizer.get_lang_id("en"))
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)



# translate yoruba to English 
yoruba_text="Itọju ilera jẹ aaye pataki ni o fẹrẹ to gbogbo awujọ nitori pe o taara ni ilera ati didara igbesi aye eniyan kọọkan. O ni awọn iṣẹ lọpọlọpọ ati awọn oojọ, pẹlu itọju idena, iwadii aisan, itọju, ati iṣakoso awọn arun ati awọn ipo. Awọn eto ilera ti o munadoko ṣe ifọkansi lati ni ilọsiwaju awọn abajade ilera, dinku iṣẹlẹ ti aisan, ati rii daju pe awọn eniyan kọọkan ni iraye si awọn iṣẹ iṣoogun pataki."
tokenizer.src_lang = "yo"
tokenizer.tgt_lang = "en"
encoded_yo = tokenizer(yoruba_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_yo, forced_bos_token_id=tokenizer.get_lang_id("en"))
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

# translate Hausa to English
hausa_text="Kiwon lafiya fage ne mai mahimmanci a kusan kowace al'umma domin yana shafar jin daɗi da ingancin rayuwar ɗaiɗaikun kai tsaye. Ya ƙunshi nau'ikan ayyuka da sana'o'i, gami da kulawa na rigakafi, ganewar asali, jiyya, da kula da cututtuka da yanayi. Ingantattun tsarin kiwon lafiya na nufin inganta sakamakon kiwon lafiya, rage yawan kamuwa da cututtuka, da kuma tabbatar da cewa mutane sun sami damar yin amfani da ayyukan likita masu mahimmanci."
tokenizer.src_lang = "ha"
tokenizer.tgt_lang = "en"
encoded_ha = tokenizer(hausa_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_ha, forced_bos_token_id=tokenizer.get_lang_id("en"))
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
```

### Supported Language Codes
- **English**: `en`
- **Yoruba**: `yo`
- **Igbo**: `ig`
- **Hausa**: `ha`

All languages supported by the [base model](https://huggingface.co/facebook/m2m100_418M) are also supported, but the performance might be below par for those languages.


### Training Dataset

The [training dataset](https://huggingface.co/datasets/HelpMum-Personal/nigeria_translation) consists of 1,500,000 translation pairs, sourced from a combination of open-source parallel corpora and curated datasets specific to Yoruba, Igbo, and Hausa

## Limitations

- While the model performs well across Yoruba, Igbo, and Hausa to English translations, performance may vary depending on the complexity and domain of the text.
- Translation quality may decrease for extremely long sentences or ambiguous contexts.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 64
- eval_batch_size: 64
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1
- mixed_precision_training: Native AMP
- 
### Framework versions

- Transformers 4.44.2
- Pytorch 2.4.0+cu121
- Datasets 2.21.0
- Tokenizers 0.19.1
