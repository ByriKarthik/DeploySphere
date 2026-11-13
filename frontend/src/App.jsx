import { useEffect, useRef, useState } from 'react'
import {
  cloneProject,
  detectProject,
  fetchHealth,
  fetchProjects,
  fetchStartupDiagnostics,
  registerProject,
} from './services/api.js'
import DeploymentPanel from './DeploymentPanel.jsx'

function StatusItem({ label, value, tone }) {
  return (
    <div className="flex items-center justify-between gap-6 border-t border-white/10 py-5 first:border-t-0 first:pt-0 last:pb-0">
      <span className="text-sm text-slate-400">{label}</span>
      <span className="flex items-center gap-2 text-sm font-medium text-slate-100">
        <span className={`h-2 w-2 rounded-full ${tone}`} aria-hidden="true" />
        {value}
      </span>
    </div>
  )
}

function getRequestError(error, fallback) {
  const detail = error.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg
  if (error.response?.status) return `${fallback} (HTTP ${error.response.status})`
  return fallback
}

function App() {
  const [health, setHealth] = useState(null)
  const [isLoadingHealth, setIsLoadingHealth] = useState(true)
  const [healthError, setHealthError] = useState('')
  const [startupDiagnostics, setStartupDiagnostics] = useState(null)
  const [isLoadingStartupDiagnostics, setIsLoadingStartupDiagnostics] = useState(true)
  const [startupDiagnosticsError, setStartupDiagnosticsError] = useState('')
  const [projects, setProjects] = useState([])
  const [isLoadingProjects, setIsLoadingProjects] = useState(true)
  const [projectsError, setProjectsError] = useState('')
  const [form, setForm] = useState({ name: '', repository_url: '', branch: 'main' })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [formError, setFormError] = useState('')
  const [formMessage, setFormMessage] = useState('')
  const [cloningProjectIds, setCloningProjectIds] = useState(() => new Set())
  const [detectingProjectIds, setDetectingProjectIds] = useState(() => new Set())
  const detectingRequests = useRef(new Set())
  const [detectionErrors, setDetectionErrors] = useState({})
  const [deploymentProjectIds, setDeploymentProjectIds] = useState(() => new Set())

  useEffect(() => {
    async function loadHealth() {
      try {
        setHealth(await fetchHealth())
      } catch (requestError) {
        if (requestError.response?.data) setHealth(requestError.response.data)
        setHealthError(getRequestError(requestError, 'Could not reach the DeploySphere API.'))
      } finally {
        setIsLoadingHealth(false)
      }
    }

    async function loadProjects() {
      try {
        setProjects(await fetchProjects())
      } catch (requestError) {
        setProjectsError(getRequestError(requestError, 'Could not load registered projects.'))
      } finally {
        setIsLoadingProjects(false)
      }
    }

    async function loadStartupDiagnostics() {
      try {
        setStartupDiagnostics(await fetchStartupDiagnostics())
      } catch {
        setStartupDiagnosticsError('Startup inspection snapshot is unavailable.')
      } finally {
        setIsLoadingStartupDiagnostics(false)
      }
    }

    loadHealth()
    loadProjects()
    loadStartupDiagnostics()
  }, [])

  async function handleSubmit(event) {
    event.preventDefault()
    setFormError('')
    setFormMessage('')

    if (!/^https:\/\/github\.com\/[A-Za-z0-9-]+\/[A-Za-z0-9._-]+(?:\.git)?\/?$/i.test(form.repository_url.trim())) {
      setFormError('Enter a GitHub repository URL such as https://github.com/owner/repository.')
      return
    }

    setIsSubmitting(true)
    try {
      const project = await registerProject({
        name: form.name,
        repository_url: form.repository_url,
        branch: form.branch,
      })
      setProjects((currentProjects) => [project, ...currentProjects])
      setForm({ name: '', repository_url: '', branch: 'main' })
      setFormMessage(`${project.name} was registered successfully.`)
    } catch (requestError) {
      setFormError(getRequestError(requestError, 'Could not register this project.'))
    } finally {
      setIsSubmitting(false)
    }
  }

  async function handleClone(project) {
    setCloningProjectIds((currentIds) => new Set(currentIds).add(project.id))
    setProjects((currentProjects) => currentProjects.map((item) => (
      item.id === project.id
        ? {
            ...item,
            workspace: {
              ...item.workspace,
              status: 'cloning',
              repository_url: item.repository_url,
              branch: item.branch,
            },
          }
        : item
    )))

    try {
      const workspace = await cloneProject(project.id)
      setProjects((currentProjects) => currentProjects.map((item) => (
        item.id === project.id ? { ...item, workspace } : item
      )))
    } catch (requestError) {
      const message = getRequestError(requestError, 'The repository could not be cloned.')
      try {
        // Reload persisted state so a 409 from another tab still shows
        // "cloning" or "cloned" instead of incorrectly showing a failure.
        setProjects(await fetchProjects())
      } catch {
        setProjects((currentProjects) => currentProjects.map((item) => (
          item.id === project.id
            ? {
                ...item,
                workspace: {
                  ...item.workspace,
                  status: 'failed',
                  repository_url: item.repository_url,
                  branch: item.branch,
                  error: { message },
                },
              }
            : item
        )))
      }
    } finally {
      setCloningProjectIds((currentIds) => {
        const nextIds = new Set(currentIds)
        nextIds.delete(project.id)
        return nextIds
      })
    }
  }

  async function handleDetect(project) {
    // The ref prevents two clicks in the same render from starting two requests.
    if (detectingRequests.current.has(project.id)) return
    detectingRequests.current.add(project.id)
    setDetectingProjectIds((currentIds) => new Set(currentIds).add(project.id))
    setDetectionErrors((currentErrors) => {
      const nextErrors = { ...currentErrors }
      delete nextErrors[project.id]
      return nextErrors
    })

    try {
      const detection = await detectProject(project.id)
      setProjects((currentProjects) => currentProjects.map((item) => (
        item.id === project.id ? { ...item, detection } : item
      )))
    } catch (requestError) {
      const message = getRequestError(requestError, 'Application detection could not be completed.')
      setDetectionErrors((currentErrors) => ({ ...currentErrors, [project.id]: message }))

      // Detection failures are persisted by the API. Reload the list so the
      // visible status comes from MongoDB and remains after a browser refresh.
      try {
        setProjects(await fetchProjects())
      } catch {
        // Keep the request error visible if MongoDB or the API is still unavailable.
      }
    } finally {
      detectingRequests.current.delete(project.id)
      setDetectingProjectIds((currentIds) => {
        const nextIds = new Set(currentIds)
        nextIds.delete(project.id)
        return nextIds
      })
    }
  }

  const apiStatus = health?.api_status ?? (healthError ? 'unavailable' : 'checking')
  const mongoStatus = health?.mongodb_status ?? (healthError ? 'unavailable' : 'checking')
  const apiTone = apiStatus === 'ok' ? 'bg-emerald-400' : 'bg-amber-400'
  const mongoTone = mongoStatus === 'connected' ? 'bg-emerald-400' : 'bg-rose-400'

  return (
    <main className="flex min-h-screen flex-col bg-slate-950 text-white">
      <header className="flex items-center justify-between border-b border-white/10 px-6 py-5 sm:px-10">
        <a href="/" className="text-xl font-bold tracking-tight text-violet-300">
          DeploySphere
        </a>
        <span className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-400">
          Application platform
        </span>
      </header>

      <section className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-6 py-14 sm:px-10 sm:py-16">
        <p className="mb-4 text-sm font-semibold uppercase tracking-[0.2em] text-violet-300">
          Deployment management
        </p>
        <h1 className="max-w-2xl text-4xl font-bold tracking-tight sm:text-6xl">
          Ship with confidence.
        </h1>
        <p className="mt-5 max-w-xl text-lg leading-8 text-slate-400">
          Register a GitHub repository and keep an eye on your platform status.
        </p>

        <section
          aria-labelledby="health-heading"
          className="mt-10 max-w-xl rounded-2xl border border-white/10 bg-slate-900/80 p-6 shadow-2xl shadow-black/20 sm:p-8"
        >
          <div className="mb-6 flex items-start justify-between gap-4">
            <div>
              <h2 id="health-heading" className="text-lg font-semibold">System health</h2>
              <p className="mt-1 text-sm text-slate-400">Live status from the DeploySphere API</p>
            </div>
            {isLoadingHealth && (
              <span className="rounded-full bg-violet-400/10 px-3 py-1 text-xs font-medium text-violet-200">
                Checking…
              </span>
            )}
          </div>
          {healthError && (
            <p role="alert" className="mb-5 rounded-lg border border-rose-400/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
              {healthError}
            </p>
          )}
          <div aria-live="polite">
            <StatusItem label="API status" value={apiStatus} tone={apiTone} />
            <StatusItem label="MongoDB status" value={mongoStatus} tone={mongoTone} />
            <StatusItem label="Database" value={health?.database ?? '—'} tone="bg-slate-500" />
          </div>
        </section>

        <section
          aria-labelledby="startup-diagnostics-heading"
          className="mt-8 max-w-xl rounded-2xl border border-white/10 bg-slate-900/80 p-6 shadow-2xl shadow-black/20 sm:p-8"
        >
          <h2 id="startup-diagnostics-heading" className="text-lg font-semibold">
            Startup deployment inspection
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            A read-only snapshot captured when this backend started.
          </p>
          {isLoadingStartupDiagnostics && (
            <p className="mt-4 text-sm text-slate-400">Loading startup snapshot…</p>
          )}
          {startupDiagnosticsError && (
            <p role="status" className="mt-4 text-sm text-slate-400">
              {startupDiagnosticsError}
            </p>
          )}
          {startupDiagnostics && (
            <div className="mt-4 space-y-3">
              <p className="text-sm text-slate-300">
                Outcome: <span className="font-medium text-white">{startupDiagnostics.outcome}</span>
                {' · '}
                Captured {new Date(startupDiagnostics.captured_at).toLocaleString()}
              </p>
              <p className="rounded-lg border border-amber-300/20 bg-amber-300/5 px-3 py-2 text-xs text-amber-100">
                {startupDiagnostics.warning}
              </p>
              <div>
                <h3 className="text-sm font-medium text-slate-200">Classification counts</h3>
                {Object.keys(startupDiagnostics.classification_counts).length === 0 ? (
                  <p className="mt-1 text-sm text-slate-400">No deployment findings were recorded.</p>
                ) : (
                  <ul className="mt-1 space-y-1 text-sm text-slate-400">
                    {Object.entries(startupDiagnostics.classification_counts).map(([classification, count]) => (
                      <li key={classification}>
                        {classification}: {count}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              {startupDiagnostics.results.some((result) => (
                !['running_verified', 'exited_verified', 'created_verified'].includes(result.classification)
              )) && (
                <div>
                  <h3 className="text-sm font-medium text-slate-200">Unresolved findings</h3>
                  <ul className="mt-1 space-y-2">
                    {startupDiagnostics.results
                      .filter((result) => (
                        !['running_verified', 'exited_verified', 'created_verified'].includes(result.classification)
                      ))
                      .map((result, index) => (
                        <li key={`${result.deployment_id}-${index}`} className="text-sm text-amber-100">
                          Deployment {result.deployment_id}: {result.classification} — {result.reason}
                        </li>
                      ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </section>

        <section
          aria-labelledby="register-heading"
          className="mt-8 max-w-xl rounded-2xl border border-white/10 bg-slate-900/80 p-6 shadow-2xl shadow-black/20 sm:p-8"
        >
          <h2 id="register-heading" className="text-lg font-semibold">Register a project</h2>
          <p className="mt-1 text-sm text-slate-400">
            Add a GitHub repository and the branch you want to manage.
          </p>

          <form className="mt-6 space-y-5" onSubmit={handleSubmit}>
            <label className="block text-sm font-medium text-slate-300">
              Project name
              <input
                required
                maxLength={100}
                value={form.name}
                onChange={(event) => setForm({ ...form, name: event.target.value })}
                className="mt-2 w-full rounded-lg border border-white/10 bg-slate-950 px-4 py-3 text-white outline-none transition placeholder:text-slate-600 focus:border-violet-400"
                placeholder="My application"
              />
            </label>
            <label className="block text-sm font-medium text-slate-300">
              GitHub repository URL
              <input
                required
                type="url"
                maxLength={300}
                value={form.repository_url}
                onChange={(event) => setForm({ ...form, repository_url: event.target.value })}
                className="mt-2 w-full rounded-lg border border-white/10 bg-slate-950 px-4 py-3 text-white outline-none transition placeholder:text-slate-600 focus:border-violet-400"
                placeholder="https://github.com/owner/repository"
              />
            </label>
            <label className="block text-sm font-medium text-slate-300">
              Branch
              <input
                required
                maxLength={250}
                value={form.branch}
                onChange={(event) => setForm({ ...form, branch: event.target.value })}
                className="mt-2 w-full rounded-lg border border-white/10 bg-slate-950 px-4 py-3 text-white outline-none transition placeholder:text-slate-600 focus:border-violet-400"
                placeholder="main"
              />
            </label>

            {formError && <p role="alert" className="text-sm text-rose-300">{formError}</p>}
            {formMessage && <p role="status" className="text-sm text-emerald-300">{formMessage}</p>}

            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full rounded-lg bg-violet-600 px-5 py-3 font-semibold text-white transition hover:bg-violet-500 disabled:cursor-wait disabled:opacity-60"
            >
              {isSubmitting ? 'Registering…' : 'Register project'}
            </button>
          </form>
        </section>

        <section aria-labelledby="projects-heading" className="mt-10 max-w-3xl">
          <div className="flex items-end justify-between gap-4">
            <div>
              <h2 id="projects-heading" className="text-xl font-semibold">Registered projects</h2>
              <p className="mt-1 text-sm text-slate-400">Repositories connected to DeploySphere</p>
            </div>
            {!isLoadingProjects && !projectsError && (
              <span className="text-sm text-slate-500">{projects.length} total</span>
            )}
          </div>

          {isLoadingProjects && <p className="mt-5 text-sm text-slate-400">Loading projects…</p>}
          {projectsError && (
            <p role="alert" className="mt-5 rounded-lg border border-rose-400/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
              {projectsError}
            </p>
          )}
          {!isLoadingProjects && !projectsError && projects.length === 0 && (
            <p className="mt-5 rounded-xl border border-dashed border-white/10 p-6 text-sm text-slate-400">
              No projects registered yet. Add your first repository using the form above.
            </p>
          )}
          {projects.length > 0 && (
            <ul className="mt-5 space-y-3">
              {projects.map((project) => (
                <li key={project.id} className="rounded-xl border border-white/10 bg-slate-900/70 p-5">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <h3 className="font-medium text-white">{project.name}</h3>
                      <a
                        href={project.repository_url}
                        target="_blank"
                        rel="noreferrer"
                        className="mt-1 inline-block break-all text-sm text-violet-300 hover:text-violet-200"
                      >
                        {project.repository_url}
                      </a>
                    </div>
                    <span className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-300">
                      {project.branch}
                    </span>
                  </div>
                  <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-white/10 pt-4">
                    <div>
                      <p className="text-sm text-slate-300">
                        Clone status: <span className="font-medium text-white">
                          {project.workspace?.status ?? 'not cloned'}
                        </span>
                      </p>
                      {project.workspace?.commit_hash && (
                        <p className="mt-1 text-xs text-slate-500">
                          Commit {project.workspace.commit_hash.slice(0, 12)}
                        </p>
                      )}
                      {project.workspace?.error?.message && (
                        <p className="mt-1 text-sm text-rose-300">{project.workspace.error.message}</p>
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={() => handleClone(project)}
                      disabled={cloningProjectIds.has(project.id) || project.workspace?.status === 'cloned'}
                      className="rounded-lg border border-violet-400/30 bg-violet-400/10 px-4 py-2 text-sm font-medium text-violet-200 transition hover:bg-violet-400/20 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {cloningProjectIds.has(project.id)
                        ? 'Cloning…'
                        : project.workspace?.status === 'failed'
                          ? 'Retry clone'
                          : project.workspace?.status === 'cloned'
                            ? 'Cloned'
                            : 'Clone Repository'}
                    </button>
                  </div>
                  <div className="mt-5 border-t border-white/10 pt-4">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <p className="text-sm text-slate-300">
                          Detection status:{' '}
                          <span className={`font-medium ${
                            (detectingProjectIds.has(project.id) || project.detection?.status === 'detecting')
                              ? 'text-violet-200'
                              : project.detection?.status === 'completed'
                                ? 'text-emerald-300'
                                : project.detection?.status === 'failed' || detectionErrors[project.id]
                                  ? 'text-rose-300'
                                  : 'text-slate-400'
                          }`}>
                            {detectingProjectIds.has(project.id)
                              ? 'detecting'
                              : project.detection?.status ?? (detectionErrors[project.id] ? 'failed' : 'not detected for current commit')}
                          </span>
                        </p>
                        {project.detection && project.detection.status !== 'detecting' && (
                          <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-sm text-slate-400">
                            <span>
                              Application: <span className="text-slate-200">
                                {project.detection.application_type === 'nodejs'
                                  ? 'Node.js'
                                  : project.detection.application_type}
                              </span>
                            </span>
                            <span>
                              Build method: <span className="text-slate-200">
                                {project.detection.build_method}
                              </span>
                            </span>
                            <span>
                              Confidence: <span className="text-slate-200">
                                {Math.round(project.detection.confidence * 100)}%
                              </span>
                            </span>
                          </div>
                        )}
                      </div>
                      {project.workspace?.status === 'cloned' && (
                        <button
                          type="button"
                          onClick={() => handleDetect(project)}
                          disabled={
                            detectingProjectIds.has(project.id)
                            || project.detection?.status === 'detecting'
                          }
                          className="rounded-lg border border-cyan-300/30 bg-cyan-300/10 px-4 py-2 text-sm font-medium text-cyan-100 transition hover:bg-cyan-300/20 disabled:cursor-wait disabled:opacity-50"
                        >
                          {detectingProjectIds.has(project.id)
                            ? 'Detecting…'
                            : project.detection?.status === 'detecting'
                              ? 'Detection in progress'
                              : project.detection?.status === 'failed'
                                ? 'Retry detection'
                                : project.detection?.status === 'completed'
                                  ? 'Detect again'
                                  : 'Detect application'}
                        </button>
                      )}
                    </div>
                    {detectionErrors[project.id] && (
                      <p role="alert" className="mt-3 rounded-lg border border-rose-400/20 bg-rose-400/10 px-3 py-2 text-sm text-rose-200">
                        {detectionErrors[project.id]}
                      </p>
                    )}
                    {project.detection?.status === 'completed' && (
                      <details className="mt-4 rounded-lg border border-white/10 bg-slate-950/50 p-3">
                        <summary className="cursor-pointer text-sm font-medium text-slate-300">
                          Detected files and warnings
                        </summary>
                        <div className="mt-3 grid gap-4 sm:grid-cols-2">
                          <div>
                            <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                              Detected files
                            </h4>
                            {project.detection.detected_files.length > 0 ? (
                              <ul className="mt-2 space-y-1 text-sm text-slate-300">
                                {project.detection.detected_files.map((file) => (
                                  <li key={file} className="break-all font-mono text-xs">{file}</li>
                                ))}
                              </ul>
                            ) : <p className="mt-2 text-sm text-slate-500">No indicator files found.</p>}
                          </div>
                          <div>
                            <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                              Warnings
                            </h4>
                            {project.detection.warnings.length > 0 ? (
                              <ul className="mt-2 space-y-1 text-sm text-amber-200/80">
                                {project.detection.warnings.map((warning) => (
                                  <li key={warning}>{warning}</li>
                                ))}
                              </ul>
                            ) : <p className="mt-2 text-sm text-slate-500">No warnings.</p>}
                          </div>
                        </div>
                      </details>
                    )}
                    {project.detection?.status === 'failed' && (
                      <p className="mt-3 text-sm text-rose-300">
                        Detection failed. Review the request message above or retry the scan.
                      </p>
                    )}
                    <div className="mt-5 border-t border-white/10 pt-4">
                      <button
                        type="button"
                        aria-expanded={deploymentProjectIds.has(project.id)}
                        onClick={() => setDeploymentProjectIds((currentIds) => {
                          const nextIds = new Set(currentIds)
                          if (nextIds.has(project.id)) nextIds.delete(project.id)
                          else nextIds.add(project.id)
                          return nextIds
                        })}
                        className="rounded-lg border border-violet-400/30 bg-violet-400/10 px-4 py-2 text-sm font-medium text-violet-200 transition hover:bg-violet-400/20"
                      >
                        {deploymentProjectIds.has(project.id) ? 'Hide deployments' : 'Manage deployments'}
                      </button>
                      {deploymentProjectIds.has(project.id) && (
                        <DeploymentPanel project={project} />
                      )}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>
      </section>

      <footer className="border-t border-white/10 px-6 py-5 text-center text-sm text-slate-500">
        DeploySphere · Application deployment platform
      </footer>
    </main>
  )
}

export default App
