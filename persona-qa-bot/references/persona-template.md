# Persona template

Personas are the core of the tool: each one drives the app with a distinct
goal and behavior, so a bug that trips many personas scores higher reach.
Define at least 2. `persona_runner.js` reads an array of these as JSON.

```json
[
  {
    "name": "New mobile user",
    "goal": "Sign up and reach the main screen on a phone",
    "viewport": { "width": 390, "height": 844 },
    "steps": [
      { "action": "goto", "url": "/" },
      { "action": "click", "selector": "text=Sign up" },
      { "action": "fill", "selector": "#email", "value": "qa+new@example.com" },
      { "action": "fill", "selector": "#password", "value": "Test1234!" },
      { "action": "click", "selector": "button[type=submit]" },
      { "action": "expectVisible", "selector": "[data-testid=home]" }
    ]
  },
  {
    "name": "Power user",
    "goal": "Perform the app's core action and confirm it saved",
    "steps": [
      { "action": "goto", "url": "/login" },
      { "action": "fill", "selector": "#email", "value": "qa+power@example.com" },
      { "action": "fill", "selector": "#password", "value": "Test1234!" },
      { "action": "click", "selector": "button[type=submit]" },
      { "action": "click", "selector": "text=New item" },
      { "action": "fill", "selector": "#item-name", "value": "sync-check-123" },
      { "action": "click", "selector": "text=Save" },
      { "action": "goto", "url": "/items" },
      { "action": "expectText", "text": "sync-check-123" }
    ]
  }
]
```

## Coverage to spread across personas

1. Core functionality (the main thing the app does)
2. Login / auth flows
3. Regressions — run the same personas against each release
4. Frontend → DB sync — did the user's action actually save? (the last two
   steps of "Power user" above check this: save, reload, assert it persisted)
5. Security flaws
6. UI breaking / layout

## Available `action` types

`goto`, `click`, `fill`, `expectText`, `expectVisible`. Add more in the
`ACTIONS` map in `persona_runner.js`.
