const { createApp, ref, computed, onMounted } = Vue

createApp({
  setup() {
    const loading = ref(true)
    const started = ref(false)
    const done = ref(false)
    const score = ref(0)
    const email = ref('')

    const cfg = ref({ webhook_url: '', webhook_secret: '' })
    const pairs = ref([])
    const order = ref([])
    const idx = ref(0)

    const leftUrl = ref('')
    const rightUrl = ref('')
    const leftType = ref('')
    const rightType = ref('')

    const validEmail = computed(() => /.+@.+\..+/.test(email.value))

    function shuffle(a){ for(let i=a.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [a[i],a[j]]=[a[j],a[i]] } return a }

    async function loadConfigAndPairs(){
      const [cfgRes, pairsRes] = await Promise.all([
        fetch('config.json').then(r=>r.json()),
        fetch('pairs.json').then(r=>r.json())
      ])
      cfg.value = cfgRes || {}
      pairs.value = (pairsRes && pairsRes.pairs) ? pairsRes.pairs : []
      order.value = shuffle([...Array(pairs.value.length).keys()])
    }

    function nextTrial(){
      if (idx.value >= order.value.length){ done.value = true; return }
      const p = pairs.value[ order.value[idx.value] ]
      const leftIsReal = Math.random() < 0.5
      leftUrl.value = leftIsReal ? p.real : p.ai
      rightUrl.value = leftIsReal ? p.ai : p.real
      leftType.value = leftIsReal ? 'real' : 'ai'
      rightType.value = leftIsReal ? 'ai' : 'real'
    }

    async function choose(side){
      const p = pairs.value[ order.value[idx.value] ]
      const isCorrect = (side === 'left' && leftType.value === 'real') || (side === 'right' && rightType.value === 'real')
      if (isCorrect) score.value++

      if (cfg.value.webhook_url) {
        try {
          await fetch(cfg.value.webhook_url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              secret: cfg.value.webhook_secret || '',
              participant_id: email.value || 'anon',
              pair_id: p.id,
              left_url: leftUrl.value,
              right_url: rightUrl.value,
              left_type: leftType.value,
              right_type: rightType.value,
              choice: side === 'left' ? 'Left' : 'Right',
              result: isCorrect ? 'T' : 'F'
            })
          })
        } catch (e) {
          console.warn('webhook failed', e)
        }
      }

      idx.value++; nextTrial()
    }

    function start(){ if (!validEmail.value) return; started.value = true; nextTrial() }

    onMounted(async () => { await loadConfigAndPairs(); loading.value = false })

    return { loading, started, done, score, email, validEmail,
             pairs, order, idx, leftUrl, rightUrl, leftType, rightType,
             start, choose }
  }
}).mount('#app')
