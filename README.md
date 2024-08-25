# AI Price Negotiator Chatbot

This repository contains an AI-driven chatbot that automates price negotiations using Natural Language Processing (NLP) techniques. The project is built using Python, Flask, and RASA, leveraging machine learning models and NLP for understanding and generating negotiation dialogues.

## Project Overview

The AI Price Negotiator Chatbot can interact with users to negotiate product prices in a human-like manner. The bot understands user intentions, responds dynamically, and optimizes its negotiation strategies based on the analyzed patterns.

### Key Features

- **Natural Language Understanding:** Uses RASA for understanding user intents and extracting key entities during the conversation.
- **Negotiation Strategy:** Implements machine learning algorithms to predict optimal counter-offers and negotiation strategies.
- **Web Interface:** Integrates a user-friendly web interface using Flask for managing negotiation sessions and enhancing user experience.
- **Customizable Dialogues:** Easily extendable dialogue management system allowing for scenario-based responses.
- **Data-Driven Learning:** Learns from past negotiations to improve the outcome of future interactions.

### Project Structure

- `models/`: Contains trained machine learning models used in negotiation strategies.
- `nlu/`: Includes intent recognition, entity extraction, and custom actions files for the RASA bot.
- `actions/`: Python scripts handling custom actions during negotiations.
- `static/` and `templates/`: Frontend assets for the Flask-based web interface.
- `rasa_core/` and `rasa_nlu/`: Core files and configurations required for RASA operations.

### How to Run the Project

1. **Clone the repository:**
    ```bash
    git clone <https://github.com/yuvjain/AI-Price-Negotiator-Chatbot/tree/main/rasa_projects>
    cd <repository-directory>
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Train the RASA model:**
    ```bash
    rasa train
    ```

4. **Start the RASA action server:**
    ```bash
    rasa run actions
    ```

5. **Run the RASA chatbot:**
    ```bash
    rasa run
    ```

6. **Start the Flask web interface:**
    ```bash
    python app.py
    ```

7. **Access the web interface:**
    Navigate to `http://localhost:5000` to interact with the bot.

### Dependencies

- Python 3.x
- RASA (Core and NLU)
- Flask
- TensorFlow
- Scikit-learn
- NLTK

### Future Improvements

- **Multi-Turn Dialogues:** Enhance the chatbot's ability to handle multi-turn conversations with more sophisticated branching.
- **Integration with E-commerce Platforms:** Connect the chatbot with e-commerce APIs to negotiate real product prices.
- **Advanced Sentiment Analysis:** Incorporate sentiment analysis to adjust negotiation strategies based on user emotions.

### Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests for any enhancements or bug fixes.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Author

Developed by Yuv Jain.  
For more projects, visit my [GitHub profile](https://github.com/yuvjain).
