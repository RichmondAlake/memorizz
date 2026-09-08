(() => {
  const state = document.getElementById('pe-state')
  if (!state) return
  const buttons = [...document.querySelectorAll('[data-action]')]
  let busy = false
  buttons.forEach(button => button.addEventListener('click', async () => {
    if (busy) return
    busy = true
    const previous = buttons.map(item => item.disabled)
    const label = button.textContent
    buttons.forEach(item => { item.disabled = true })
    button.textContent = button.dataset.action === 'reflect' ? 'Reviewing memories…' : 'Saving…'
    const error = document.getElementById('pe-error')
    const notice = document.getElementById('pe-notice')
    error.hidden = true
    notice.hidden = true
    try {
      const response = await fetch('/persona-evolution/action', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: state.dataset.userId, revision: Number(state.dataset.revision),
          proposal_id: state.dataset.proposalId, action: button.dataset.action,
          ...(button.dataset.action === 'pause' ? { paused: button.dataset.paused === 'true' } : {}),
          ...(button.dataset.action === 'schedule' ? { daily_enabled: button.dataset.dailyEnabled === 'true' } : {}),
        }),
      })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail || 'Persona action failed. Reload and try again.')
      if (result.notice === 'no_new_evidence') {
        notice.textContent = 'No new evidence since the last review. Your approach is unchanged; no model call was made.'
        notice.hidden = false
        buttons.forEach((item, index) => { item.disabled = previous[index] })
        button.textContent = label
        busy = false
        return
      }
      window.location.reload()
    } catch (err) {
      error.textContent = err.message || 'Persona action failed.'
      error.hidden = false
      buttons.forEach((item, index) => { item.disabled = previous[index] })
      button.textContent = label
      busy = false
    }
  }))
})()
