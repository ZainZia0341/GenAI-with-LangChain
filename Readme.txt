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

A U-Net is a popular type of convolutional neural network (CNN) that has a symmetric encoder-decoder structure. It is commonly used for image segmentation tasks, but in the context of Stable Diffusion, it can help to introduce and remove noise in the image. The U-Net takes a noisy image (seed) as input and processes it through a series of convolutional layers to extract features and learn semantic representations.





Title LangChain for LLM Apps

Stochastic parrots refers to LLMs that can produce convincing language but lack any true comprehension of the meaning behind words. Coined by researchers Emily Bender, Timnit Gebru, Margaret Mitchell, and Angelina McMillan-Major in their influential paper On the Dangers of Stochastic Parrots (2021), the term critiques models that mindlessly mimic linguistic patterns. Without being grounded in the real world, models can produce responses that are inaccurate, irrelevant, unethical, or make little logical sense.

Outdated knowledge: LLMs rely solely on their training data. Without external integration, they cannot provide recent real-world information.
Inability to take action: LLMs cannot perform interactive actions like searches, calculations, or lookups. This severely limits functionality.
Hallucination risks: Insufficient knowledge on certain topics can lead to the generation of incorrect or nonsensical content by LLMs if not properly grounded.
Biases and discrimination: Depending on the data they were trained on, LLMs can exhibit biases that can be religious, ideological, or political in nature.
Lack of transparency: The behavior of large, complex models can be opaque and difficult to interpret, posing challenges to alignment with human values.
Lack of context: LLMs may struggle to understand and incorporate context from previous prompts or conversations. They may not remember previously mentioned details or may fail to provide additional relevant information beyond the given prompt.

As for reasoning, for example, an LLM may correctly identify a fruit’s density and water’s density when asked about those topics independently, but it would struggle to synthesize those facts to determine if the fruit will float (this being a multi-hop question). The model fails to bridge its disjointed knowledge.

LLM apps typically have the following components:
A client layer to collect user input as text queries or decisions.
A prompt engineering layer to construct prompts that guide the LLM.
An LLM backend to analyze prompts and produce relevant text responses.
An output parsing layer to interpret LLM responses for the application interface.
Optional integration with external services via function APIs, knowledge bases, and reasoning algorithms to augment the LLM’s capabilities.

Created in 2022 by Harrison Chase, LangChain is an open-source Python framework for building LLM-powered applications. It provides developers with modular, easy-to-use components for connecting language models with external data sources and services. The project has attracted millions in venture capital funding from the likes of Sequoia Capital and Benchmark, who supplied funding to Apple, Cisco, Google, WeWork, Dropbox, and many other successful companies.

As for the broader ecosystem, LangSmith is a platform that complements LangChain by providing robust debugging, testing, and monitoring capabilities for LLM applications. For example, developers can quickly debug new chains by viewing detailed execution traces. Alternative prompts and LLMs can be evaluated against datasets to ensure quality and consistency. Usage analytics empower data-driven decisions around optimizations.

LlamaHub and LangChainHub provide open libraries of reusable elements to build sophisticated LLM systems in a simplified manner. LlamaHub is a library of data loaders, readers, and tools created by the LlamaIndex community. It provides utilities to easily connect LLMs to diverse knowledge sources. The loaders ingest data for retrieval, while tools enable models to read/write to external data services. LlamaHub simplifies the creation of customized data agents to unlock LLM capabilities.

LangChainHub is a central repository for sharing artifacts like prompts, chains, and agents used in LangChain. Inspired by the Hugging Face Hub, it aims to be a one-stop resource for discovering high-quality building blocks to compose complex LLM apps. The initial launch focuses on a collection of reusable prompts. Future plans involve adding support for chains, agents, and other key LangChain components.

LangFlow and Flowise are UIs that allow chaining LangChain components in an executable flowchart by dragging sidebar components onto the canvas and connecting them together to create your pipeline. This is a quick way to experiment and prototype pipelines

LangChain and LangFlow can be deployed locally, for example, using the Chainlit library, or on different platforms, including Google Cloud. The langchain-serve library helps to deploy both LangChain and LangFlow on the Jina AI cloud as LLM-apps-as-a-service with a single command.

Chains are a critical concept in LangChain for composing modular components into reusable pipelines. For example, developers can put together multiple LLM calls and other components in a sequence to create complex applications for things like chatbot-like social interactions, data extraction, and data analysis. In the most generic terms, a chain is a sequence of calls to components, which can include other chains. The most innocuous example of a chain is probably the PromptTemplate, which passes a formatted response to a language model. More complex chains integrate models with tools like LLMMath, for math-related queries, or SQLDatabaseChain, for querying databases. These are called utility chains, because they combine language models with specific tools. implements chains to make sure the content of the output is not toxic, does not otherwise violate OpenAI’s moderation rules (OpenAIModerationChain), or that it conforms to ethical, legal, or custom principles (ConstitutionalChain). An LLMCheckerChain verifies statements to reduce inaccurate responses using a technique called self-reflection. The LLMCheckerChain can prevent hallucinations and reduce inaccurate responses by verifying the assumptions underlying the provided statements and questions. A few chains can make autonomous decisions. Like agents, router chains can decide which tool to use based on their descriptions. A RouterChain can dynamically select which retrieval system, such as prompts or indexes, to use.

Agents are a key concept in LangChain for creating systems that interact dynamically with users and environments over time. An agent is an autonomous software entity that is capable of taking actions to accomplish goals and tasks.

Chains and agents are similar concepts and it’s worth unpicking their differences. The core idea in LangChain is the compositionality of LLMs and other components to work together. Both chains and agents do that, but in different ways. Both extend LLMs, but agents do so by orchestrating chains while chains compose lower-level modules. While chains define reusable logic by sequencing components, agents leverage chains to take goal-driven actions. Agents combine and orchestrate chains. The agent observes the environment, decides which chain to execute based on that observation, takes the chain’s specified action, and repeats.

In LangChain, memory refers to the persisting state between executions of a chain or agent. Robust memory approaches unlock key benefits for developers building conversational and interactive applications. For example, storing chat history context in memory improves the coherence and relevance of LLM responses over time.

ConversationBufferMemory stores all messages in model history. This increases latency and costs.
ConversationBufferWindowMemory retains only recent messages.
ConversationKGMemory summarizes exchanges as a knowledge graph for integration into prompts.
EntityMemory backed by a database persists agent state and facts.

Machine translator: A language model can use a machine translator to better comprehend and process text in multiple languages. This tool enables non-translation-dedicated language models to understand and answer questions in different languages.
Calculator: Language models can utilize a simple calculator tool to solve math problems. The calculator supports basic arithmetic operations, allowing the model to accurately solve mathematical queries in datasets specifically designed for math problem-solving.
Maps: By connecting with the Bing Map API or similar services, language models can retrieve location information, assist with route planning, provide driving distance calculations, and offer details about nearby points of interest.
Weather: Weather APIs provide language models with real-time weather information for cities worldwide. Models can answer queries about current weather conditions or forecast the weather for specific locations within varying time periods.
Stocks: Connecting with stock market APIs like Alpha Vantage allows language models to query specific stock market information such as opening and closing prices, highest and lowest prices, and more.
Slides: Language models equipped with slide-making tools can create slides using high-level semantics provided by APIs such as the python-pptx library or image retrieval from the internet based on given topics. These tools facilitate tasks related to slide creation that are required in various professional fields.
Table processing: APIs built with pandas DataFrames enable language models to perform data analysis and visualization tasks on tables. By connecting to these tools, models can provide users with a more streamlined and natural experience for handling tabular data.
Knowledge graphs: Language models can query knowledge graphs using APIs that mimic human querying processes, such as finding candidate entities or relations, sending SPARQL queries, and retrieving results. These tools assist in answering questions based on factual knowledge stored in knowledge graphs.
Search engine: By utilizing search engine APIs like Bing Search, language models can interact with search engines to extract information and provide answers to real-time queries. These tools enhance the model’s ability to gather information from the web and deliver accurate responses.
Wikipedia: Language models equipped with Wikipedia search tools can search for specific entities on Wikipedia pages, look up keywords within a page, or disambiguate entities with similar names. These tools facilitate question-answering tasks using content retrieved from Wikipedia.
Online shopping: Connecting language models with online shopping tools allows them to perform actions like searching for items, loading detailed information about products, selecting item features, going through shopping pages, and making purchase decisions based on specific user instructions.





Title Getting Started with LangChain

