# QBI

## Web server

Run the frontend web server with Node.js:

```sh
cd frontend
npm start
```

By default it serves files from `frontend` at `http://localhost:3000`.

To use another port:

```sh
cd frontend
PORT=3015 npm start
```

The server is kept inside `frontend/` and should use cloud data sources rather than local `data/` folders.
