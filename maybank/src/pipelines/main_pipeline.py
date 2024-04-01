if __name__ == "__main__":
    import os
    import sys
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    project_dir = "C:/Users/samue/Downloads/Maybank/"
    os.chdir(project_dir)

    sys.path.append(project_dir)

    from maybank.src.pipelines.data_pipeline import run_data_pipeline
    from maybank.src.pipelines.model_pipeline import run_model_pipeline

    run_data_pipeline()

    run_model_pipeline()
