// All roadmap content lives here so phases/items can be edited
// without touching any component. `link: null` means the item is a
// concept/milestone rather than a specific tool — rendered as plain text.

const roadmap = [
  {
    id: "phase-1",
    title: "Python Foundations",
    dayRange: "Day 1–9",
    description: "Set up the workbench every ML engineer needs.",
    accent: "#f5c542",
    items: [
      { id: "python", label: "Python", link: "https://docs.python.org" },
      { id: "jupyter", label: "Jupyter", link: "https://jupyter.org/documentation" },
      { id: "conda", label: "Conda", link: "https://docs.conda.io" },
      { id: "venv", label: "venv", link: "https://docs.python.org/3/library/venv.html" },
    ],
  },
  {
    id: "phase-2",
    title: "Versioning & Experiment Tracking",
    dayRange: "Day 10–30",
    description: "Version data and models, and track every experiment.",
    accent: "#8b5cf6",
    items: [
      { id: "dvc", label: "DVC", link: "https://dvc.org/doc" },
      { id: "git", label: "Git", link: "https://git-scm.com/doc" },
      { id: "mlflow", label: "MLflow", link: "https://mlflow.org/docs/latest/index.html" },
      { id: "pipelines", label: "Pipelines", link: "https://dvc.org/doc/user-guide/pipelines" },
      { id: "metrics", label: "Metrics", link: null },
    ],
  },
  {
    id: "phase-3",
    title: "Training, Features & Data Quality",
    dayRange: "Day 31–49",
    description: "Train models, tune them, and keep the data honest.",
    accent: "#f97316",
    items: [
      { id: "scikit-learn", label: "scikit-learn", link: "https://scikit-learn.org/stable/documentation.html" },
      { id: "pytorch", label: "PyTorch", link: "https://pytorch.org/docs/stable/index.html" },
      { id: "optuna", label: "Optuna", link: "https://optuna.readthedocs.io" },
      { id: "mlflow", label: "MLflow", link: "https://mlflow.org/docs/latest/index.html" },
      { id: "great-expectations", label: "Great Expectations", link: "https://docs.greatexpectations.io" },
      { id: "vault", label: "HashiCorp Vault", link: "https://developer.hashicorp.com/vault/docs" },
      { id: "yaml", label: "YAML", link: "https://yaml.org/spec/" },
    ],
  },
  {
    id: "phase-4",
    title: "Packaging & Serving",
    dayRange: "Day 50–66",
    description: "Wrap models into containers and serve them over APIs.",
    accent: "#2dd4bf",
    items: [
      { id: "docker", label: "Docker", link: "https://docs.docker.com" },
      { id: "fastapi", label: "FastAPI", link: "https://fastapi.tiangolo.com" },
      { id: "mlflow-model", label: "MLflow Model", link: "https://mlflow.org/docs/latest/models.html" },
      { id: "bentoml", label: "BentoML", link: "https://docs.bentoml.com" },
      { id: "multi-stage-build", label: "Multi-stage build", link: "https://docs.docker.com/build/building/multi-stage/" },
      { id: "healthchecks", label: "Healthchecks", link: "https://docs.docker.com/reference/dockerfile/#healthcheck" },
      { id: "ci-build", label: "CI Build", link: null },
    ],
  },
  {
    id: "phase-5",
    title: "Monitoring & CI/CD",
    dayRange: "Day 67–84",
    description: "Watch models in production and automate every release.",
    accent: "#ec4899",
    items: [
      { id: "evidently", label: "Evidently AI", link: "https://docs.evidentlyai.com" },
      { id: "prometheus", label: "Prometheus", link: "https://prometheus.io/docs/introduction/overview/" },
      { id: "grafana", label: "Grafana", link: "https://grafana.com/docs" },
      { id: "drift-detection", label: "Drift Detection", link: null },
      { id: "cicd", label: "CI/CD", link: null },
      { id: "tests", label: "Tests", link: null },
      { id: "cml-reports", label: "CML Reports", link: "https://cml.dev/doc" },
      { id: "rollback", label: "Rollback", link: null },
    ],
  },
  {
    id: "phase-6",
    title: "Orchestration, Kubernetes & Capstone",
    dayRange: "Day 85–100",
    description: "Orchestrate everything on Kubernetes and ship the capstone.",
    accent: "#10b981",
    items: [
      { id: "kubernetes", label: "Kubernetes", link: "https://kubernetes.io/docs/home/" },
      { id: "argocd", label: "ArgoCD", link: "https://argo-cd.readthedocs.io" },
      { id: "gitops", label: "GitOps", link: "https://opengitops.dev" },
      { id: "kserve", label: "KServe", link: "https://kserve.github.io/website/" },
      { id: "prefect", label: "Prefect", link: "https://docs.prefect.io" },
      { id: "helm", label: "Helm", link: "https://helm.sh/docs/" },
      { id: "capstone", label: "Capstone Project", link: null },
      { id: "e2e-deploy", label: "End-to-End Deploy", link: null },
    ],
  },
];

export default roadmap;
