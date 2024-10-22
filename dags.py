import dagshub
dagshub.init(repo_owner='Kushagra-Bisht', repo_name='Smartphone_portfolio_project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
  