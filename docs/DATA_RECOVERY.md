# Recovering the dataset from the Social Engine site

The rulebook does not hand over the dataset. It has to be recovered from
<https://datavortex-social-engine.vercel.app/>. This is the path we followed and how we verified the files.

## Walkthrough

1. **Landing: Recovery Terminal.** The boot console prints `social-engine kernel v4.1.7 -- init`, then
   `data_intake.service ......... FAILED`, `database.service ... FAILED (corrupted)`, and so on.
   `[ ENTER RECOVERY MODE ]` continues.
2. **Operator authentication.** You create an OPERATOR ID and ACCESS KEY. The key stays in the browser's local storage.
3. **Dashboard.** Ten modules (HOME, DATABASE, ANALYTICS, REPORTS, USERS, LIVE SIGNAL, SYSTEM, BACKUP, ARCHIVE, HELP)
   each fail in their own way: the loader hangs, the signal drops to static, a page error appears, access is denied, or nothing responds.
   ARCHIVE fails with `ARCH_LINKLOST_07`. After a few failed clicks the page shows
   *"Nothing responding? There may be another way in."*
4. **System log (hint: "pay attention to what appears at the beginning of each line").** Each log line starts
   with a timestamp and the component reporting it:
   ```
   03:42:17  database.service        exited with code 137
   03:42:19  analytics.service       segfault, core not found
   03:42:21  live_signal.service     carrier lost, retrying...
   03:42:21  live_signal.service     retry failed, giving up
   03:42:23  node_07                 responded 200 (intermittent)
   03:42:26  watchdog                last known good node: node_07
   03:42:31  watchdog                dashboard link to node_07: severed
   ```
   Only `node_07` is alive, and its dashboard link is severed. So the node must be reached another way ("decode the connection, find the node").
5. **Recovery shell** (terminal button, bottom-right). `help` lists `help, status, scan, logs, clear`.
   `scan` reports *"node_07 responding intermittently, outside normal routing. Manual reconnection may be possible."*
   `logs` prints the log above and pins *"last known surviving node: node_07"*.
6. **`connect node_07`** (also accepted: `reconnect`, `link`, `access`, `restore` + `node_07`/`archive`) prints
   `establishing manual link to node_07... connection stabilized. redirecting to archive interface...`
7. **ARCHIVE NODE 07** → *SURVIVING DATA SOURCES FOUND* → **DATASET 01 · DEPLOYED** → `[ RECOVER ]` both files:
   - `Social_Engine_Users.csv`: 1,500 rows × 5 columns
   - `Social_Engine_Posts_Corrupted.csv`: 12,360 rows × 8 columns

## Verification

- The site creates both CSVs **in the browser**. Their contents ship inside the JavaScript bundle and are saved
  through a `Blob` download, so the Network tab shows no separate CSV request.
- We compared our downloads with the strings embedded in the bundle. They are **byte-identical**:

  | File | Bytes | SHA-256 |
  |---|---|---|
  | `Social_Engine_Posts_Corrupted.csv` | 2,042,090 | `49a460d6b6d8b7f5bbbc1605852e555611adffdc9d7248c2d33910e2dd5dce42` |
  | `Social_Engine_Users.csv` | 76,897 | `d30efe470a4dd5612266c3b3ebcc5e84528ef12dc5d832698a6e2a64283f7fd8` |

- `src/clean.py` pins these hashes and records `matches_recovered_archive` in `reports/cleaning_audit.json`.
- **Line-ending trap:** with Git's default `core.autocrlf=true` on Windows, checked-out CSVs gain `\r\n`
  (including inside quoted multi-line text), which changes the hash. `.gitattributes` marks `*.csv -text`
  so a clone on any OS reproduces the exact bytes.
