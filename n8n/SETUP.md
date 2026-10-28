# Phase 6 - Automatisation avec n8n

## Ce que fait le workflow

```
Receive request (Webhook POST)
   |
   v
Analyze message  --(API en panne)--> Hand over to a human
   |
   v
Needs a human?
   |-- oui --> Record escalation --> Notify the human agent
   |
   `-- non --> Prepare review --> Wait for human approval (formulaire)
                                      |
                                      v
                                  Approved? --oui--> Save to simulated outbox
                                      |
                                      `--non--> Rejected - nothing sent
```

**Règle d'or :** rien n'est envoyé à un client. Le dernier nœud écrit seulement
dans `data/runtime/outbox_simulation.jsonl`, et uniquement après qu'un humain
a rempli le formulaire (décision + son nom).

## 1. Démarrer l'API de l'agent

Dans le dossier du projet (environnement virtuel activé) :

```powershell
pip install -r requirements.txt
python -m uvicorn src.api:create_app --factory --host 127.0.0.1 --port 8000
```

Vérification : ouvre http://127.0.0.1:8000/health, tu dois voir `{"status":"ok",...}`.
Test de l'API seule, avant n8n :
```powershell
$body = @{ message = "How long does standard delivery take?"; history = @() } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/analyze -ContentType "application/json" -Body $body
```
Tu dois voir `route : await_human_approval`.
La documentation interactive est sur http://127.0.0.1:8000/docs.

## 2. Installer n8n

**Option A - Docker Desktop (le plus simple sous Windows)**
```powershell
docker run -it --rm --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n
```
Dans ce cas, n8n tourne dans un conteneur : dans les 3 nœuds « HTTP Request »
(Analyze message, Record escalation, Save to simulated outbox) remplace `http://127.0.0.1:8000` par `http://host.docker.internal:8000`.

**Option B - sans Docker**
Il faut une version récente de Node.js (vérifie la version exigée sur docs.n8n.io, puis `node -v`) :
```powershell
npx n8n
```

Ouvre ensuite http://localhost:5678 et crée ton compte local.

## 3. Importer le workflow

n8n > **Workflows** > menu `...` > **Import from file** > choisis
`n8n/support_agent_workflow.json`. Le workflow est importé **désactivé**.

## 4. Tester

1. Dans le workflow, clique sur **Execute workflow** (n8n écoute l'URL de test).
2. Envoie une demande depuis PowerShell :
```powershell
$body = @{ message = "How long does standard delivery take?"; history = @() } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:5678/webhook-test/support-request -ContentType "application/json" -Body $body
```
3. Le workflow s'arrête au nœud **Wait for human approval** (statut « Waiting »).
   Ouvre l'exécution, clique sur le nœud **Prepare review** et copie la valeur
   `review_form_url` : c'est le lien **signé** du formulaire. Ouvre-le dans ton navigateur,
   choisis une décision, saisis ton nom, envoie. (Un lien écrit à la main est refusé par n8n.)
4. Regarde `data/runtime/outbox_simulation.jsonl` : la réponse approuvée y est
   écrite avec `"sent": false`.

Cas sensible (pas de formulaire, l'escalade passe par un humain) :
```powershell
$body = @{ message = "Someone hacked my account"; history = @() } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:5678/webhook-test/support-request -ContentType "application/json" -Body $body
```
Le cas est consigné dans `data/runtime/escalations.jsonl`.

Conversation (le mécontentement qui dure est détecté grâce à `history`) :
```powershell
$body = @{ message = "Still nothing and this is terrible"; history = @("My order ORD-77001 is late, this is unacceptable") } | ConvertTo-Json
```

## 5. Passer en « production »

Active le workflow (interrupteur **Active**), puis utilise l'URL
`http://localhost:5678/webhook/support-request` (sans `-test`).

## 6. Clé API (recommandée si n8n est dans Docker)

1. Dans `.env`, ajoute `SUPPORT_API_KEY=une-longue-valeur-aleatoire`.
2. Redémarre l'API.
3. Dans les 3 nœuds « HTTP Request » du workflow, remplace la valeur de l'en-tête
   `X-API-Key` par cette même valeur.

Sans clé configurée, l'API accepte les appels locaux sans en-tête.
N'écoute jamais sur `0.0.0.0` sans clé API.

## 7. Brancher un vrai canal (plus tard)

- Remplace **Receive request** par un déclencheur Gmail, Slack ou Telegram
  (il faut créer tes identifiants dans n8n ; ils ne sont jamais stockés dans ce fichier).
- Remplace **Notify the human agent** par un nœud d'e-mail ou de messagerie pour
  alerter tes agents.
- Avant **Wait for human approval**, ajoute un nœud qui envoie le lien du
  formulaire (`{{ $json.review_form_url }}`, déjà préparé par **Prepare review**) à tes relecteurs.
- Ne remplace **Save to simulated outbox** par un vrai envoi qu'à la fin, et
  **garde l'étape d'approbation**.

## Dépannage

| Symptôme | Cause probable |
|---|---|
| Le flux part dans « Hand over to a human » | L'API est arrêtée, la clé API est fausse, ou le message est vide. Regarde l'erreur dans l'exécution. |
| `ECONNREFUSED` | L'API n'est pas lancée, ou n8n est dans Docker (utilise `host.docker.internal`). |
| « Invalid Form Link » | Tu as ouvert un lien écrit à la main. Utilise `review_form_url` (lien signé) du nœud **Prepare review**. |
| 404 sur le webhook | En mode test, clique d'abord sur « Execute workflow ». En production, active le workflow. |
| Erreur 422 à la dernière étape | Tu as choisi « Approve with my edited reply » sans écrire de réponse modifiée. |
