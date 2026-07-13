# Part 4 — LLM-Powered Feature: Model Prediction Explanation Pipeline (Track C)

## Steps to run
Download llm_integration.ipynb and best_model.pkl from part 4 folder of repo. Open the llm_integration.ipynb in google collab. Add the best_model.pkl in the files of collab. We are using gemini as llm for explanation, so please add key in serect and key name should be LLM_API_KEY and run it step by step.

## Track Details
Track C which is model prediction explanation pipeline has been selected for Part 4 of the Capstone project. This pipeline integrates our machine learning model best_model.pkl with the Google Gemini API to explain individual churn predictions.


### Libraries, Configuration, and JSON Schema Definition
We load all essential Python libraries required as below
- pandas and numpy for data structures.
- joblib for deserializing the model.
- requests for making HTTP POST requests to the LLM API.
- jsonschema for validating the JSON response.

We ues EXPLANATION_SCHEMA, a Python dictionary that maps the expected output schema with exactly 5 required fields, those are prediction_label which is string, confidence_level is also string but restricted to enum "low" or "medium" or "high",top_reason which is also an string,second_reason is als0 an string, next_step is an string.

## 1. Reusable LLM Call Function
**call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512)** is implemented as resuable LLM call function.
This function dynamically detects if it is running inside Google Colab and loads LLM_API_KEY from Google Colab Secrets using google.colab.userdata.get or running locally loading from a local .env file. Submits an HTTP POST request and parses the raw text content from the JSON response structure.

### Simple Connection Verification Test
A test prompt is dispatched to verify the API key connection.
- **System Prompt**: `You are a helpful assistant. Reply with only the word: hello`
- **User Prompt**: `hello`
- **Response**: `hello`
This confirms that the API key is fetched from secret, URL configurations and payload parser are working correctly.

## 2. PII Guardrails Check
To protect user privacy, we have pattern check in the input
    **email_pattern** = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    **phone_pattern** = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'

- **PII Test Input**: `"Customer email is sivarajan@test.com. Churn predicted: Yes."` -> **Blocked** (Prints `"Input blocked: PII detected."` and returns `None`).
- **Clean Test Input**: `"Customer on Contract: Month-to-month, tenure: 2 months."` -> **Passed** (Proceeds to the LLM query).

## 3. Feature Preprocessing for Machine Learning Model
We write encode_record to align the raw customer feature dictionary with the columns and categories used to train best_model.pkl:
- Ordinal mapping of Contract and InternetService.
- One-hot encoding of variables like gender, PaymentMethod, etc.
- Column re-indexing and padding with zeros for missing features.
- Conversion of boolean values to binary integers (`0` or `1`).

## 4. Load ML Model & Setup Prompts
Load the model pipeline best_model.pkl using joblib.load() and extract the feature_names_in_ attribute. Three test profiles representing different contract types, tenures, and charges has been setup. We also define the verbatim system prompt.

## 5. Temperature A/B Comparison
By executing the explanation call twice for Customer Profile with `temperature=0.0` and once with `temperature=0.7`to demonstrate the effect of temperature on deterministic vs creative token selection.

### 6. End-to-End Batch Pipeline
We loop through all three customer profiles: preprocess features, make predictions and probabilities using the ML model, construct the user prompt, perform PII checks, request the LLM explanation, strip formatting, parse/validate the response against the schema, and gracefully apply a null-filled dictionary fallback if validation fails.


## Verbatim Prompts

### System Prompt
```text
You are an expert customer churn analyst. Explain the machine learning model's churn prediction for a specific customer based on their feature values, predicted class, and probability. You must output ONLY a valid JSON object matching this schema:
{
  "prediction_label": "A short string summarizing the risk status",
  "confidence_level": "low|medium|high",
  "top_reason": "Primary driver behind this model prediction",
  "second_reason": "Secondary contributing factor to the prediction",
  "next_step": "Recommended action for customer retention or service maintenance"
}
Do not include any markdown formatting, backticks, or text outside the JSON object. Output ONLY valid JSON.
```

### User Prompt Template
```text
Customer details:
- Contract type: {Contract}
- Internet service: {InternetService}
- Tenure duration: {tenure} months
- Total Charges: ${TotalCharges}
- Payment Method: {PaymentMethod}

Model Churn Prediction: Class {predicted_class}
Model Churn Probability: {predicted_prob}
```

---

## Temperature Choice & Rationale

When set `temperature = 0.0` for our production pipeline.
- **Produces deterministic outputs**: Large language models predict the next token by calculating a probability distribution over the vocabulary. When temperature is set to `0.0`, the model performs **greedy decoding**, which means it always chooses the token with the absolute highest probability. This guarantees that calling the API with the exact same inputs will produce the exact same output.

When set `temperature = 0.7` for our production pipeline.
- **Produces variability**: Increasing the temperature to `0.7` scales the logic before applying softmax, softening the probability distribution. This allows the model to sample from a wider distribution of tokens, introducing vocabulary diversity, synonyms, and variations in sentence structures.


## Temperature A/B Comparison Table (Customer Profile 1)

| Input Profile | Output at temp = 0.0 | Output at temp = 0.7 | Key Difference |
| :--- | :--- | :--- | :--- |
| **Customer 1**<br>(Month-to-month, Fiber optic, tenure=2, Total=$170.0, Pred=1, Prob=0.7075) | `{"prediction_label":"High Churn Risk","confidence_level":"high","top_reason":"The customer has a very short tenure of only two months on a month-to-month contract.","second_reason":"The customer pays for high-cost Fiber optic service using a manual electronic check.","next_step":"Proactively offer a discounted long-term contract and incentivize switching to automatic payments."}` | `{"prediction_label":"High Churn Risk","confidence_level":"high","top_reason":"The customer has an extremely short tenure of only two months on a volatile month-to-month contract.","second_reason":"The customer uses electronic check payments for high-cost fiber optic internet service.","next_step":"Offer an incentive to transition to a long-term contract with automated billing."}` | **temp=0.0** uses concise and consistent phrasing. **temp=0.7** introduces descriptive variations and adds minor reasoning details due to broader token sampling. |


## Structured Output Validation Flow

To ensure program robustness, our pipeline implements a structured validation for responses:
1. **Whitespace Cleaning**: We strip any leading or trailing spaces from the LLM's raw text response.
2. **Markdown Stripping**: If the LLM returns code-block markdown wrappers (` ```json ` and ` ``` `), we strip them away.
3. **JSON Decoding**: We attempt to parse the cleaned string into a Python dictionary using `json.loads()`. We wrap this inside a `try-except json.JSONDecodeError` block to catch syntax errors.
4. **Schema Enforcement**: We validate the parsed dictionary structure, keys, values, and enums against EXPLANATION_SCHEMA using `jsonschema.validate()`. We catch errors inside a `try-except jsonschema.ValidationError` block.
5. **Graceful Fallback**: If any stage fails, we log the specific error details and return a fallback dictionary where all required fields are set to **None** except next_step:
   ```python
   {
       "prediction_label": None,
       "confidence_level": None,
       "top_reason": None,
       "second_reason": None,
       "next_step": "Fallback applied. Error message: {error_details}"
   }
   ```


## End-to-End Demonstration Table

The following table summarizes the end-to-end run on the three handcrafted profiles:

| Profile Input Details | Predicted Class | Probability | Explanation JSON Output | Valid JSON | Pass/Block (Guardrail) |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **Customer 1**:<br>- Month-to-month<br>- Fiber optic<br>- Tenure: 2 months<br>- Total: $170.00 | `1` (Churn) | `0.7075` | `{"prediction_label":"High Churn Risk","confidence_level":"high","top_reason":"The customer has a very short tenure of only two months on a month-to-month contract.","second_reason":"The customer pays for high-cost Fiber optic service using a manual electronic check.","next_step":"Proactively offer a discounted long-term contract and incentivize switching to automatic payments."}` | **Pass** | **Passed** (No PII) |
| **Customer 2**:<br>- Two year<br>- DSL<br>- Tenure: 60 months<br>- Total: $1200.00 | `0` (Stay) | `0.0461` | `{"prediction_label":"Very Low Risk","confidence_level":"high","top_reason":"The customer has a long tenure of 60 months and is committed to a stable two-year contract.","second_reason":"Low total charges and reliable DSL service further minimize the likelihood of churn.","next_step":"Propose transitioning to paperless billing or autopay to streamline their payment process."}` | **Pass** | **Passed** (No PII) |
| **Customer 3**:<br>- Month-to-month<br>- DSL<br>- Tenure: 12 months<br>- Total: $480.00 | `0` (Stay) | `0.2284` | `{"prediction_label":"Low Churn Risk","confidence_level":"high","top_reason":"The customer has established a stable 12-month tenure using lower-risk DSL internet service.","second_reason":"Consistent and moderate total spending of $480.0 indicates steady service utilization.","next_step":"Offer an incentive to transition from month-to-month to a one-year contract to secure long-term loyalty."}` | **Pass** | **Passed** (No PII) |
