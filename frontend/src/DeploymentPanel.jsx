import { useCallback, useEffect, useRef, useState } from 'react'
import {
  buildProjectImage,
  createProjectDeployment,
  fetchActiveDeployment,
  fetchProjectBuilds,
  fetchProjectDeployments,
  restartProjectDeployment,
  stopProjectDeployment,
} from './services/api.js'

const PAGE_SIZE = 10

function getRequestError(error, fallback) {
  const detail = error.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg
  if (error.response?.status) return `${fallback} (HTTP ${error.response.status})`
  return fallback
}

function createIdempotencyKey() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID()
  if (globalThis.crypto?.getRandomValues) {
    const bytes = globalThis.crypto.getRandomValues(new Uint8Array(16))
    return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('')
  }
  throw new Error('Secure random values are unavailable; deployment was not submitted.')
}

async function fetchDeploymentSnapshot(projectId, skip) {
  const [history, active] = await Promise.all([
    fetchProjectDeployments(projectId, skip, PAGE_SIZE),
    fetchActiveDeployment(projectId).catch((error) => {
      if (error.response?.status === 404) return null
      throw error
    }),
  ])
  return { history, active }
}

function DeploymentSummary({ deployment, title }) {
  if (!deployment) return null

  return (
    <div className="mt-3 rounded-lg border border-white/10 bg-slate-950/60 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="text-sm font-medium text-slate-200">{title}</h4>
        <span className="rounded-full border border-white/10 px-2.5 py-1 text-xs text-slate-300">
          {deployment.status}
        </span>
      </div>
      {deployment.deployment_url && deployment.status === 'running' && (
        <a
          href={deployment.deployment_url}
          target="_blank"
          rel="noreferrer"
          className="mt-2 inline-block break-all text-sm text-violet-300 hover:text-violet-200"
        >
          {deployment.deployment_url}
        </a>
      )}
      {deployment.error?.message && (
        <p className="mt-2 text-sm text-rose-300">{deployment.error.message}</p>
      )}
    </div>
  )
}

export default function DeploymentPanel({ project }) {
  const [builds, setBuilds] = useState([])
  const [selectedBuildId, setSelectedBuildId] = useState('')
  const [containerPort, setContainerPort] = useState(
    String(project.deployment_config?.port ?? 8080),
  )
  const [buildsLoading, setBuildsLoading] = useState(true)
  const [buildsError, setBuildsError] = useState('')
  const [isBuilding, setIsBuilding] = useState(false)
  const [buildMessage, setBuildMessage] = useState('')
  const buildingRequest = useRef(false)
  const [deployments, setDeployments] = useState([])
  const [activeDeployment, setActiveDeployment] = useState(null)
  const [deploymentSkip, setDeploymentSkip] = useState(0)
  const [deploymentsLoading, setDeploymentsLoading] = useState(true)
  const [deploymentsError, setDeploymentsError] = useState('')
  const [isDeploying, setIsDeploying] = useState(false)
  const [lifecycleDeploymentId, setLifecycleDeploymentId] = useState('')
  const [actionError, setActionError] = useState('')
  const [actionMessage, setActionMessage] = useState('')
  const [portError, setPortError] = useState('')
  const pendingDeploymentRequest = useRef(null)

  const requestBuilds = useCallback(
    () => fetchProjectBuilds(project.id),
    [project.id],
  )

  const loadBuilds = useCallback(async () => {
    setBuildsLoading(true)
    setBuildsError('')
    try {
      const result = await requestBuilds()
      setBuilds(result)
    } catch (error) {
      setBuildsError(getRequestError(error, 'Could not load image builds.'))
    } finally {
      setBuildsLoading(false)
    }
  }, [requestBuilds])

  const loadDeployments = useCallback(async (skip) => {
    setDeploymentsLoading(true)
    setDeploymentsError('')
    try {
      const { history, active } = await fetchDeploymentSnapshot(project.id, skip)
      setDeployments(history)
      setActiveDeployment(active)
      setDeploymentSkip(skip)
    } catch (error) {
      setDeploymentsError(getRequestError(error, 'Could not load deployment information.'))
    } finally {
      setDeploymentsLoading(false)
    }
  }, [project.id])

  useEffect(() => {
    let isCurrent = true
    requestBuilds()
      .then((result) => {
        if (isCurrent) setBuilds(result)
      })
      .catch((error) => {
        if (isCurrent) {
          setBuildsError(getRequestError(error, 'Could not load image builds.'))
        }
      })
      .finally(() => {
        if (isCurrent) setBuildsLoading(false)
      })
    fetchDeploymentSnapshot(project.id, 0)
      .then(({ history, active }) => {
        if (!isCurrent) return
        setDeployments(history)
        setActiveDeployment(active)
        setDeploymentSkip(0)
      })
      .catch((error) => {
        if (isCurrent) {
          setDeploymentsError(getRequestError(error, 'Could not load deployment information.'))
        }
      })
      .finally(() => {
        if (isCurrent) setDeploymentsLoading(false)
      })
    return () => {
      isCurrent = false
    }
  }, [loadDeployments, project.id, requestBuilds])

  const successfulBuilds = builds.filter((build) => build.status === 'succeeded')
  const detection = project.detection
  const canBuildImage = detection?.status === 'completed'
    && ['python', 'nodejs'].includes(detection.application_type)
    && detection.build_method === 'dockerfile'
  const parsedPort = /^\d+$/.test(containerPort) ? Number(containerPort) : NaN
  const validPort = Number.isInteger(parsedPort) && parsedPort >= 1 && parsedPort <= 65535

  async function handleBuildImage() {
    if (buildingRequest.current || !canBuildImage) return
    buildingRequest.current = true
    setIsBuilding(true)
    setBuildsError('')
    setBuildMessage('')
    try {
      const build = await buildProjectImage(project.id)
      if (build.status === 'succeeded') {
        setBuildMessage(
          build.image_id
            ? `Image build succeeded (${build.image_id}).`
            : 'Image build succeeded.',
        )
      } else {
        setBuildsError(
          build.error?.message || `Image build completed with status: ${build.status}.`,
        )
      }
    } catch (error) {
      setBuildsError(getRequestError(error, 'Could not build the project image.'))
    } finally {
      await loadBuilds()
      buildingRequest.current = false
      setIsBuilding(false)
    }
  }

  async function handleDeploy(event) {
    event.preventDefault()
    setActionError('')
    setActionMessage('')
    setPortError('')

    if (!validPort) {
      setPortError('Enter a whole application container port from 1 to 65535.')
      return
    }
    if (!successfulBuilds.some((build) => build.id === selectedBuildId)) {
      setActionError('Select a successful image build before deploying.')
      return
    }

    const requestFingerprint = JSON.stringify([
      project.id,
      selectedBuildId,
      parsedPort,
    ])
    let idempotencyKey = pendingDeploymentRequest.current?.fingerprint === requestFingerprint
      ? pendingDeploymentRequest.current.key
      : null
    if (!idempotencyKey) {
      try {
        idempotencyKey = createIdempotencyKey()
      } catch (error) {
        setActionError(error.message)
        return
      }
      pendingDeploymentRequest.current = {
        fingerprint: requestFingerprint,
        key: idempotencyKey,
      }
    }

    setIsDeploying(true)
    try {
      const deployment = await createProjectDeployment(
        project.id,
        { image_build_id: selectedBuildId, container_port: parsedPort },
        idempotencyKey,
      )
      if (pendingDeploymentRequest.current?.key === idempotencyKey) {
        pendingDeploymentRequest.current = null
      }
      setActionMessage(`Deployment request confirmed with status: ${deployment.status}.`)
      await loadDeployments(0)
    } catch (error) {
      setActionError(getRequestError(error, 'The deployment request could not be completed.'))
    } finally {
      setIsDeploying(false)
    }
  }

  async function handleLifecycleAction(deployment, action) {
    setLifecycleDeploymentId(deployment.id)
    setActionError('')
    setActionMessage('')
    try {
      const updated = action === 'stop'
        ? await stopProjectDeployment(project.id, deployment.id)
        : await restartProjectDeployment(project.id, deployment.id)
      setActionMessage(`Deployment ${action} confirmed with status: ${updated.status}.`)
      await loadDeployments(deploymentSkip)
    } catch (error) {
      setActionError(getRequestError(error, `Could not ${action} this deployment.`))
      await loadDeployments(deploymentSkip)
    } finally {
      setLifecycleDeploymentId('')
    }
  }

  const hasNextPage = deployments.length === PAGE_SIZE

  return (
    <section aria-label={`Deployments for ${project.name}`} className="mt-5 border-t border-white/10 pt-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h4 className="font-medium text-white">Deployments</h4>
          <p className="mt-1 text-sm text-slate-400">
            Choose a successful image build and its application port.
          </p>
        </div>
        <button
          type="button"
          onClick={() => loadDeployments(deploymentSkip)}
          disabled={deploymentsLoading}
          className="rounded-lg border border-white/10 px-3 py-2 text-sm text-slate-300 transition hover:bg-white/5 disabled:cursor-wait disabled:opacity-50"
        >
          {deploymentsLoading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      <DeploymentSummary deployment={activeDeployment} title="Active deployment" />
      {activeDeployment && (
        <div className="mt-3 flex flex-wrap gap-2">
          {activeDeployment.status === 'running' && (
            <button
              type="button"
              onClick={() => handleLifecycleAction(activeDeployment, 'stop')}
              disabled={Boolean(lifecycleDeploymentId)}
              className="rounded-lg border border-amber-300/30 bg-amber-300/10 px-3 py-2 text-sm font-medium text-amber-100 hover:bg-amber-300/20 disabled:opacity-50"
            >
              {lifecycleDeploymentId === activeDeployment.id ? 'Stopping…' : 'Stop'}
            </button>
          )}
          {['running', 'stopped'].includes(activeDeployment.status) && (
            <button
              type="button"
              onClick={() => handleLifecycleAction(activeDeployment, 'restart')}
              disabled={Boolean(lifecycleDeploymentId)}
              className="rounded-lg border border-cyan-300/30 bg-cyan-300/10 px-3 py-2 text-sm font-medium text-cyan-100 hover:bg-cyan-300/20 disabled:opacity-50"
            >
              {lifecycleDeploymentId === activeDeployment.id ? 'Restarting…' : 'Restart'}
            </button>
          )}
        </div>
      )}
      {!activeDeployment && !deploymentsLoading && !deploymentsError && (
        <p className="mt-3 text-sm text-slate-500">No active deployment.</p>
      )}

      <form onSubmit={handleDeploy} className="mt-4 grid gap-3 rounded-lg border border-white/10 bg-slate-950/40 p-4 sm:grid-cols-[minmax(0,1fr)_auto_10rem_auto] sm:items-end">
        <label className="block text-sm font-medium text-slate-300">
          Successful image build
          <select
            required
            value={selectedBuildId}
            onChange={(event) => {
              pendingDeploymentRequest.current = null
              setSelectedBuildId(event.target.value)
            }}
            disabled={buildsLoading || successfulBuilds.length === 0 || isDeploying}
            className="mt-2 w-full rounded-lg border border-white/10 bg-slate-900 px-3 py-2.5 text-white outline-none focus:border-violet-400 disabled:opacity-60"
          >
            <option value="">
              {buildsLoading ? 'Loading builds…' : successfulBuilds.length ? 'Select a build' : 'No successful builds'}
            </option>
            {successfulBuilds.map((build) => (
              <option key={build.id} value={build.id}>
                {build.commit_hash.slice(0, 12)} · {build.dockerfile_path} · {new Date(build.created_at).toLocaleString()}
              </option>
            ))}
          </select>
        </label>
        <div>
          <button
            type="button"
            onClick={handleBuildImage}
            disabled={!canBuildImage || isBuilding}
            className="rounded-lg border border-cyan-300/30 bg-cyan-300/10 px-4 py-2.5 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-300/20 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isBuilding ? 'Building image…' : 'Build Image'}
          </button>
          {!canBuildImage && (
            <p className="mt-2 text-xs text-slate-500">
              Complete detection for a supported application with a Dockerfile before building.
            </p>
          )}
        </div>
        <label className="block text-sm font-medium text-slate-300">
          Container port
          <input
            type="number"
            required
            min="1"
            max="65535"
            step="1"
            value={containerPort}
            onChange={(event) => {
              pendingDeploymentRequest.current = null
              setContainerPort(event.target.value)
            }}
            aria-invalid={Boolean(portError)}
            className="mt-2 w-full rounded-lg border border-white/10 bg-slate-900 px-3 py-2.5 text-white outline-none focus:border-violet-400"
          />
        </label>
        <button
          type="submit"
          disabled={isDeploying || buildsLoading || successfulBuilds.length === 0}
          className="rounded-lg bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-violet-500 disabled:cursor-wait disabled:opacity-50"
        >
          {isDeploying ? 'Deploying…' : 'Deploy'}
        </button>
      </form>

      {buildsLoading && <p className="mt-3 text-sm text-slate-400">Loading image builds…</p>}
      {buildsError && <p role="alert" className="mt-3 text-sm text-rose-300">{buildsError}</p>}
      {buildMessage && <p role="status" className="mt-3 text-sm text-emerald-300">{buildMessage}</p>}
      {!buildsLoading && !buildsError && builds.length === 0 && (
        <p className="mt-3 text-sm text-slate-500">No image builds found for this project.</p>
      )}
      {portError && <p role="alert" className="mt-3 text-sm text-rose-300">{portError}</p>}
      {actionError && <p role="alert" className="mt-3 text-sm text-rose-300">{actionError}</p>}
      {actionMessage && <p role="status" className="mt-3 text-sm text-emerald-300">{actionMessage}</p>}
      {deploymentsError && <p role="alert" className="mt-3 text-sm text-rose-300">{deploymentsError}</p>}

      <div className="mt-5">
        <h4 className="text-sm font-medium text-slate-200">Deployment history</h4>
        {deploymentsLoading && <p className="mt-2 text-sm text-slate-400">Loading deployment history…</p>}
        {!deploymentsLoading && deployments.length === 0 && !deploymentsError && (
          <p className="mt-2 text-sm text-slate-500">No deployment history yet.</p>
        )}
        {!deploymentsLoading && deployments.length > 0 && (
          <ul className="mt-2 space-y-2">
            {deployments.map((deployment) => (
              <li key={deployment.id} className="rounded-lg border border-white/10 bg-slate-950/40 p-3">
                <DeploymentSummary deployment={deployment} title={`Deployment ${deployment.id.slice(0, 8)}`} />
                <p className="mt-2 text-xs text-slate-500">
                  {deployment.commit_hash ? `Commit ${deployment.commit_hash.slice(0, 12)} · ` : ''}
                  {new Date(deployment.created_at).toLocaleString()}
                </p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {deployment.status === 'running' && (
                    <button
                      type="button"
                      onClick={() => handleLifecycleAction(deployment, 'stop')}
                      disabled={Boolean(lifecycleDeploymentId)}
                      className="rounded-md border border-amber-300/30 px-2.5 py-1 text-xs text-amber-100 hover:bg-amber-300/10 disabled:opacity-50"
                    >
                      Stop
                    </button>
                  )}
                  {['running', 'stopped'].includes(deployment.status) && (
                    <button
                      type="button"
                      onClick={() => handleLifecycleAction(deployment, 'restart')}
                      disabled={Boolean(lifecycleDeploymentId)}
                      className="rounded-md border border-cyan-300/30 px-2.5 py-1 text-xs text-cyan-100 hover:bg-cyan-300/10 disabled:opacity-50"
                    >
                      Restart
                    </button>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
        <div className="mt-3 flex items-center justify-between gap-3">
          <button
            type="button"
            onClick={() => loadDeployments(Math.max(0, deploymentSkip - PAGE_SIZE))}
            disabled={deploymentSkip === 0 || deploymentsLoading}
            className="rounded-lg border border-white/10 px-3 py-2 text-sm text-slate-300 hover:bg-white/5 disabled:opacity-40"
          >
            Previous
          </button>
          <span className="text-xs text-slate-500">Page {Math.floor(deploymentSkip / PAGE_SIZE) + 1}</span>
          <button
            type="button"
            onClick={() => loadDeployments(deploymentSkip + PAGE_SIZE)}
            disabled={!hasNextPage || deploymentsLoading || deploymentSkip >= 10_000}
            className="rounded-lg border border-white/10 px-3 py-2 text-sm text-slate-300 hover:bg-white/5 disabled:opacity-40"
          >
            Next
          </button>
        </div>
      </div>
    </section>
  )
}
