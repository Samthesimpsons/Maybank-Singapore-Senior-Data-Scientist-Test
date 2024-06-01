# Bank Customer Segmentation

My attempt landed me the next stage of interview for Senior DS Role, which I decline to proceed further.

## Business Problem Overview

The bank aims to increase its customer base and revenue by identifying potential affluent customers within its Existing To Bank (ETB) segment. By upgrading these customers from normal to affluent status, the bank can offer tailored products and services to enhance satisfaction and drive revenue growth.

### Target Segment

Focus on the Existing To Bank (ETB) customers, particularly those currently classified as normal but with the potential to become affluent.

### Data Utilization

Analyze a comprehensive dataset featuring various customer attributes and a binary label indicating 'affluent' or 'normal' status to identify potential candidates for upgrade.

### Expected Result

The bank is poised to effectively identify and target hidden affluent customers within the ETB segment, leading to increased revenue, customer satisfaction, and a stronger market presence.

## Machine Learning Problem Definition:

### Objective

- **Customer Segmentation:** Utilize supervised machine learning (classification) to identify potential affluent customers within the ETB segment.
- **Identification of Upgrade Candidates:** Predict whether a customer should be classified as affluent or normal based on their data profile, focusing on maximizing recall for the affluent class. False positives would actually be the hidden affluent customers to target.

### Data Overview

- **Features:** Customer data from the provided dataset, including demographic information, transaction history, account balances, etc.
- **Labels Usage:** Binary labels indicating whether each customer is affluent or normal, used for training and evaluating the classification model.

### Model Development

- **Preprocessing and Feature Engineering:** Clean and preprocess data to ensure it's suitable for classification, including handling missing values, encoding categorical variables, etc.
- **Model Selection:** Choose appropriate classification algorithms (e.g., XGBoost, Random Forest) to predict the target variable effectively.
- **Model Evaluation:** Given the best model choice, tune the model with a focus on maximizing recall for the affluent class.

### Business Application

- **Targeted Marketing:** Utilize predictions from the classification model to target potential affluent customers with tailored marketing campaigns and product offerings.
- **Segment Analysis:** Analyze the characteristics and behaviors of predicted affluent customers to refine marketing strategies and enhance customer engagement.

### Rationale for Classification over Clustering

While clustering can provide valuable insights into grouping customers based on similarities in their data profiles, the decision to opt for classification instead was driven by several key factors:
- **Granular Predictions:** Classification provides individual predictions for each customer, enabling precise targeting and personalized strategies.
  
- **Interpretability:** Classification models offer feature importance metrics, helping understand the factors driving segmentation.

- **Threshold Control:** With classification, the bank can set thresholds for predicting affluent customers, aligning with strategic goals.

- **Model Evaluation:** Clear evaluation metrics like recall allow for measuring model effectiveness and refining approaches.

- **Strategy Development:** Insights from classification aid in developing targeted marketing and product strategies.

## Technical Details

1. **Folder Structure:**
- The current folder structure is similar to Kedro, making it easy for MLE to deploy into an actual Kedro project.
   - `maybank/conf` contains all the yaml configurations. `maybank/conf/local` is empty as no secret credentials such as S3 keys are being used.
   - `maybank/data` contains all the saved data (raw, processed), and also the html for the EDA analysis.
   - `maybank/docs` contains the sphinx documentation for the `maybanks/src` folder.
   - `maybank/notebooks/main_notebook.ipynb` contains the data processing, EDA and modeling work.
   - `maybank/src/pipelines/main_pipeline.py` runs the entire pipeline from end-to-end before inference stage.
   - `maybank/notebooks/analysis.ipynb` contains the inferencing and analysis of results.
   - `maybank/src` contains all the scripts that the notebook imports.
   - `maybank/tests` is empty at the moment, it is for pytest integration, mirrored to `maybanks/src` folder.
   

1. **Dependency Management**: 
   - I used `Poetry` for managing project dependencies.
   - It provides a reliable and efficient tool for dependency management.
   - Steps to install dependencies:
      ```bash
      pip install poetry

      # Inside directory of pyproject.toml
      poetry install

      # Optional to work within the virtual env that poetry automatically creates
      # Else inside the notebooks just need to activate the virtual env created similar to any virtual env
      poetry shell
      ```

2. **Linting**: 
   - I have implemented linting with `ruff` and `flake8` to ensure code consistency and quality, handled by `Poetry`.
   - `Ruff` is lightning fast due to it's `rust` implementation.
   - **ENSURE** `Make` is installed first. If not, you can use the bash script.
   - Steps to run lint:
      ```bash
      # Using Make
      make lint

      # Using Bash
      bash lint.sh
      ```

3. **Documentation**:
   - I have set up sphinx documentation, to see my sphinx configurations, look under `docs/config.py`.
   - To view the documentation, look under `docs/html/index.html` to view the entire interactive HTML documentation.
   - Steps to rerun docs:
      ```bash
      cd docs

      # Using Make
      make clean
      make html
      ```

## Deployment Process Using Kedro with MLflow and Airflow

### Overview
Here is a quick rundown and approach I would use for this machine learning project:

1. Develop with `Kedro` and `Poetry`
2. Integrate `MLflow` for experiment tracking and workflow
3. Build a `Docker` file to package the entire project
4. Set up CI/CD pipelines with `Jenkins`
5. Deploy the project to `OpenShift`
6. Create a `Django` REST API project to expose my deployed project endpoint
7. Integrate backend calls to a front-end for the bank users to quickly drop a dataset of **ETB Customers** for quick inference
8. Set up `OpenTelemetry` and integrate it with a UI such as `Kibana` for logging and tracing of deployed model inference API calls

### More detailed steps

1. **Develop with Kedro and Poetry:**
   - Organize this machine learning project using `Kedro` for project structuring and workflow management.
   - Use `Poetry` for dependency management, ensuring consistent environments across different machines.

2. **Integrate MLflow for Experiment Tracking and Workflow:**
   - Incorporate `MLflow` into this `Kedro` project for experiment tracking, model versioning, and workflow management.

3. **Build a Dockerfile to Package Entire Project:**
   - Create a `Dockerfile` to package this `Kedro` project, `MLflow`, and other dependencies into a containerized environment.

4. **Setting up CI/CD Pipelines with Jenkins:**
   - Write a `Jenkinsfile` defining the CI/CD pipeline for this project.
   - Include stages for checking out the source code, building the `Docker` image, running tests and linting, pushing the image to a container registry, and deploying to `OpenShift`.

5. **Deploy Project to OpenShift:**
   - Set up an `OpenShift` cluster and create a project for deploying this machine learning project.
   - Prepare a deployment configuration file (deployment.yaml) describing how to deploy this Dockerized project in `OpenShift`.
   - Apply the deployment configuration to the `OpenShift` project.

6. **Create a Django REST API Project:**
   - Develop a `Django` REST API project to expose endpoints for the deployed machine learning model.
   - Implement endpoints for inference, allowing users to send data for predictions.

7. **Integrate Backend Call to a Front-End:**
   - Develop a front-end interface for bank users to interact with the `Django` REST API.
   - Allow users to upload datasets of ETB customers for quick inference using the deployed machine learning model.

8. **Set up OpenTelemetry and Integrate with UI such as Kibana:**
   - Implement `OpenTelemetry` for logging and tracing of deployed model inference API calls.
   - Integrate `OpenTelemetry` with a UI tool like `Kibana` to visualize and analyze logs and traces, enabling efficient monitoring and debugging.

### Additional Considerations

- **PySpark Decision:**
  - The decision not to use `PySpark` for a small dataset but planning for its deployment later is a strategic choice balancing current efficiency with future scalability.
  - There will just be a need to refactor the `Pandas` codes to `PySpark`, which is relatively simple.
