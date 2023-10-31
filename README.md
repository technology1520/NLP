# NLP Text Analysis Project

This project involves various Natural Language Processing (NLP) tasks to analyze a text, which can be in a TXT format. Below, you'll find a step-by-step guide on how to use and understand this project.

Project Overview
In this project, we perform comprehensive text analysis on a provided text document. The primary steps include:

Text Preprocessing: We remove section or chapter names, pictures, and tables using regular expressions. This ensures we focus on the textual content for analysis.

Tokenization: We break the text into tokens and remove common stop words to prepare the text for further analysis.

Token Frequency Distribution: We analyze the frequency distribution of tokens to understand the most common words in the text.

Word Cloud Creation: We create a word cloud to visually represent the distribution of words in the text.

Part-of-Speech (PoS) Tagging: We apply PoS tagging to understand the grammatical structure of the text, using one of the four tag sets studied in our class.

Bi-gram Probability Table: We create a bi-gram probability table for the largest chapter, keeping stop words to understand word associations within the chapter.

Fill-in-the-Blanks Game: In a different chapter, we use the bi-gram probability table to play the fill-in-the-blanks game and assess the accuracy of our predictions against the original sentence.

Requirements: Make sure you have the required Python libraries installed. You may need to install libraries like NLTK, regular expressions, and word cloud.

Input Text: Place your text file (in TXT format) in the project directory. Ensure the file is named input.txt.

Running the Analysis: Execute the main analysis script, which will perform all the steps mentioned above. Use the following command:

Copy code
python analyze_text.py
Results: The analysis results will be displayed in the console and stored in output files.

Comparing Accuracy: To assess the accuracy of the fill-in-the-blanks game, check the output file and compare the generated sentence with the original.

Example Outputs
You can find example outputs and analysis results in the output/ directory within this repository.

License
This project is under the MIT License, allowing for open use and modification.

Credits
This project was created as a part of an NLP course and is maintained by Arin Khandelwal. If you have any questions or suggestions, please feel free to reach out at arinkhandelwal44@gmail.com
