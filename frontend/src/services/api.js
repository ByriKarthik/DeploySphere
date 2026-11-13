import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001'

/** Request the current API and MongoDB health status from FastAPI. */
export async function fetchHealth() {
  const response = await axios.get(`${API_BASE_URL}/api/health`)
  return response.data
}

/** Load the retained, historical startup deployment-inspection snapshot. */
export async function fetchStartupDiagnostics() {
  const response = await axios.get(`${API_BASE_URL}/api/diagnostics/startup-inspection`)
  return response.data
}

/** Register a GitHub repository with the DeploySphere API. */
export async function registerProject(project) {
  const response = await axios.post(`${API_BASE_URL}/api/projects`, project)
  return response.data
}

/** Load all projects that have been registered with DeploySphere. */
export async function fetchProjects() {
  const response = await axios.get(`${API_BASE_URL}/api/projects`)
  return response.data
}

/** Clone the saved repository and branch for one registered project. */
export async function cloneProject(projectId) {
  const response = await axios.post(`${API_BASE_URL}/api/projects/${projectId}/clone`)
  return response.data
}

/** Run application detection for a previously cloned project. */
export async function detectProject(projectId) {
  const response = await axios.post(`${API_BASE_URL}/api/projects/${projectId}/detect`)
  return response.data
}

/** Load the saved detection details for one project when a fresh read is needed. */
export async function fetchProjectDetection(projectId) {
  const response = await axios.get(`${API_BASE_URL}/api/projects/${projectId}/detection`)
  return response.data
}

/** Build the current successfully detected Dockerfile for one project. */
export async function buildProjectImage(projectId) {
  const response = await axios.post(`${API_BASE_URL}/api/projects/${projectId}/build`)
  return response.data
}

/** Load the most recent image builds for one project. */
export async function fetchProjectBuilds(projectId) {
  const response = await axios.get(`${API_BASE_URL}/api/projects/${projectId}/builds`, {
    params: { skip: 0, limit: 100 },
  })
  return response.data
}

/** Create a deployment from an explicit image build and container port. */
export async function createProjectDeployment(projectId, deployment, idempotencyKey) {
  const response = await axios.post(
    `${API_BASE_URL}/api/projects/${projectId}/deployments`,
    deployment,
    { headers: { 'Idempotency-Key': idempotencyKey } },
  )
  return response.data
}

/** Load one page of deployment history for a project. */
export async function fetchProjectDeployments(projectId, skip = 0, limit = 10) {
  const response = await axios.get(`${API_BASE_URL}/api/projects/${projectId}/deployments`, {
    params: { skip, limit },
  })
  return response.data
}

/** Load the active deployment for a project. */
export async function fetchActiveDeployment(projectId) {
  const response = await axios.get(`${API_BASE_URL}/api/projects/${projectId}/deployments/active`)
  return response.data
}

/** Stop one project-owned deployment. */
export async function stopProjectDeployment(projectId, deploymentId) {
  const response = await axios.post(
    `${API_BASE_URL}/api/projects/${projectId}/deployments/${deploymentId}/stop`,
  )
  return response.data
}

/** Restart one project-owned deployment. */
export async function restartProjectDeployment(projectId, deploymentId) {
  const response = await axios.post(
    `${API_BASE_URL}/api/projects/${projectId}/deployments/${deploymentId}/restart`,
  )
  return response.data
}
