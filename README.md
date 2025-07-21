# RAG-Pipeline
Retrieval-Augmented Generation (RAG) using indexed documents in order to evaluate input text with Deepseek-R1 8B Model.
> The Deepseek-R1 8B model requires 11GB of GPU to be hosted.
The documents used were public Game Design Documents (GDDs). They were found from the following source: [GAMESCRYE Game Design Documents](https://gamescrye.com/resources/game-design-documents/)
The best GDDs were converted into vector space and subsequent raw form along with indexed locations were obtained. These were used to augment the prompt that is fed to the Large Language Model (LLM).
> Edit the input text file with the data to be evaluated to manipulated. The prompt text file should be edited with instructions on achieving the required task.
