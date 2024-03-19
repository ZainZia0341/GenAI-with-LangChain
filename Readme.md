Unit ttile What is Generative AI


Zero-shot means the models were prompted with the question, while in 5-shot settings, models were additionally given 5 question-answer examples. These added examples could naively account for about 20% of performance according to Measuring Massive Multitask Language Understanding (Hendrycks and colleagues, revised 2023).

There are a few differences between these models and their training that can account for these boosts in performance, such as scale, instruction-tuning, a tweak to the attention mechanisms, and more and different training data. First and foremost, the massive scaling in parameters from 1.5 billion (GPT-2) to 175 billion (GPT-3) to more than a trillion (GPT-4) enables models to learn more complex patterns; however, another major change in early 2022 was the post-training fine-tuning of models based on human instructions, which teaches the model how to perform a task by providing demonstrations and feedback.

Artificial Intelligence (AI) is a broad field of computer science focused on creating intelligent agents that can reason, learn, and act autonomously.

Machine Learning (ML) is a subset of AI focused on developing algorithms that can learn from data.

Deep Learning (DL) uses deep neural networks, which have many layers, as a mechanism for ML algorithms to learn complex patterns from data.

Generative Models are a type of ML model that can generate new data based on patterns learned from input data.

Language Models (LMs) are statistical models used to predict words in a sequence of natural language. Some language models utilize deep learning and are trained on massive datasets, becoming large language models (LLMs).

Text-to-text: Models that generate text from input text, like conversational agents. Examples: LLaMa 2, GPT-4, Claude, and PaLM 2.
Text-to-image: Models that generate images from text captions. Examples: DALL-E 2, Stable Diffusion, and Imagen.
Text-to-audio: Models that generate audio clips and music from text. Examples: Jukebox, AudioLM, and MusicGen.
Text-to-video: Models that generate video content from text descriptions. Example: Phenaki and Emu Video.
Text-to-speech: Models that synthesize speech audio from input text. Examples: WaveNet and Tacotron.
Speech-to-text: Models that transcribe speech to text [also called Automatic Speech Recognition (ASR)]. Examples: Whisper and SpeechGPT.
Image-to-text: Models that generate image captions from images. Examples: CLIP and DALL-E 3.
Image-to-image: Applications for this type of model are data augmentation such as super-resolution, style transfer, and inpainting.
Text-to-code: Models that generate programming code from text. Examples: Stable Diffusion and DALL-E 3.
Video-to-audio: Models that analyze video and generate matching audio. Example: Soundify.

As mentioned, the availability of cheaper and more powerful hardware has been a key factor in the development of deeper models. This is because DL models require a lot of computing power to train and run. This concerns all aspects of processing power, memory, and disk space. This graph shows the cost of computer storage over time for different mediums such as disks, solid state, flash, and internal memory in terms of price in dollars per terabyte (adapted from Our World in Data by Max Roser, Hannah Ritchie, and Edouard Mathieu; 
https://ourworldindata.org/grapher/historical-cost-of-computer-memory-and-storage:

The importance of the number of parameters in an LLM: The more parameters a model has, the higher its capacity to capture relationships between words and phrases as knowledge. As a simple example of these higher-order correlations, an LLM could learn that the word “cat” is more likely to be followed by the word “dog” if it is preceded by the word “chase,” even if there are other words in between. Generally, the lower a model’s perplexity, the better it will perform, for example, in terms of answering questions.

GPUs are particularly well suited for the matrix/vector computations necessary to train deep learning neural networks, therefore significantly increasing the speed and efficiency of these systems by several orders of magnitude and reducing running times from weeks to days.

Transformer models, introduced in 2017, built upon this progress and enabled the creation of large-scale models like GPT-3. Transformers rely on attention mechanisms and resulted in a further leap in the performance of generative models. These models, such as Google’s BERT and OpenAI’s GPT series, can generate highly coherent and contextually relevant text.

The development of transfer learning techniques, which allow a model pre-trained on one task to be fine-tuned on another, similar task, has also been significant. These techniques have made it more efficient and practical to train large generative models. Moreover, part of the rise of generative models can be attributed to the development of software libraries and tools (TensorFlow, PyTorch, and Keras) specifically designed to work with these artificial neural networks, streamlining the process of building, training, and deploying them.

Representation learning is about a model learning its internal representations of raw data to perform a machine learning task, rather than relying only on engineered feature extraction. For example, an image classification model based on representation learning might learn to represent images according to visual features like edges, shapes, and textures. The model isn’t told explicitly what features to look for – it learns representations of the raw pixel data that help it make predictions.

Despite the remarkable achievements, language models still face limitations when dealing with complex mathematical or logical reasoning tasks. It remains uncertain whether continually increasing the scale of language models will inevitably lead to new reasoning capabilities. Further, LLMs are known to return the most probable answers within the context, which can sometimes yield fabricated information, termed hallucinations. This is a feature as well as a bug since it highlights their creative potential.

A transformer is a DL architecture, first introduced in 2017 by researchers at Google and the University of Toronto (in an article called Attention Is All You Need; Vaswani and colleagues), that comprises self-attention and feed-forward neural networks, allowing it to effectively capture the word relationships in a sentence. The attention mechanism enables the model to focus on various parts of the input sequence.

The size of the data points indicates training cost in terms of petaFLOPs and petaFLOP/s-days. A petaFLOP/s day is a unit of throughput that consists of performing 10 to the power of 15 operations per day. Training operations in the calculations are estimated as the approximate number of addition and multiplication operations based on the GPU utilization efficiency.

A foundation model (sometimes known as a base model) is a large model that was trained on an immense quantity of data at scale so that it can be adapted to a wide range of downstream tasks. In GPT models, this pre-training is done via self-supervised learning.

Other notable foundational GPT models besides OpenAI’s include Google DeepMind’s PaLM 2, the model behind Google’s chatbot Bard. Although GPT-4 leads most benchmarks in performance, these and other models demonstrate a comparable performance in some tasks and have contributed to advancements in generative transformer-based language models.

The releases of the LLaMa and LLaMa 2 series of models, with up to 70B parameters, by Meta AI in February and July 2023, respectively, have been highly influential by enabling the community to build on top of them, thereby kicking off a Cambrian explosion of open-source LLMs. LLaMa triggered the creation of models such as Vicuna, Koala, RedPajama, MPT, Alpaca, and Gorilla. LLaMa 2, since its recent release, has already inspired several very competitive coding models, such as WizardCoder.

Claude and Claude 2 are AI assistants created by Anthropic. Evaluations suggest Claude 2, released in July 2023, is one of the best GPT-4 competitors in the market. It improves on previous versions in helpfulness, honesty, and lack of stereotype bias based on human feedback comparisons. It also performs well on standardized tests like GRE and MBE. Key model improvements include an expanded context size of up to 200K tokens, far larger than most available models, and being commercial or open source. It also performs better on use cases like coding, summarization, and long document understanding.

the goal in pre-training is to minimize perplexity, which means the model’s predictions align more with the actual outcomes.

Models like Midjourney, DALL-E 2, and Stable Diffusion provide creative and realistic images derived from textual input or other images. These models work by training deep neural networks on large datasets of image-text pairs. The key technique used is diffusion models, which start with random noise and gradually refine it into an image through repeated denoising steps.

The unique aspect of generative image models is the reverse diffusion process, where the model attempts to recover the original image from a noisy, meaningless image. By iteratively applying noise removal transformations, the model generates images of increasing resolutions that align with the given text input. The final output is an image that has been modified based on the text input.

