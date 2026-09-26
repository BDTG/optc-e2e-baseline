# Human-eval pack — SLM 0.5B giải thích (40 mẫu, BLIND — không xem key khi chấm)

## Cách chấm (người chấm: thầy / bạn cùng nhóm)
- correctness (1–5): verdict + TTP + evidence có đúng với chain không?
- usefulness (1–5): analyst đọc xong có hành động được không?
- hallucination (Y/N): có chi tiết nào BỊA (không có trong chain) không?
- Ghi vào bảng dưới. Chấm xong mới mở `human-eval-key.json` đối chiếu.

| # | nid | chain (rút gọn) | verdict | TTP | evidence | action | corr 1-5 | use 1-5 | hal Y/N | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ADGEN_0079656 | Event1 | C:\Windows\System32\bcdedit.exe | cmd: c:\windows\system32\bcdedit.exe /set {current} bootstatuspolicy ignoreallfailures | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 2 | ADGEN_0001374 | Event10 | C:\Program Files\VMware\VMware Tools\vmtoolsd.exe | Event10 | C:\Program Files\VMware\VMware Tools\vmtoolsd.exe | Event10 | C:\Program Files\VMware\VM | MAL | T1059 | parent_chain | terminate_process |  |  |  |  |
| 3 | ADGEN_0023405 | Event3004 | Event3089 | BEN | none | parent_chain | no_action |  |  |  |  |
| 4 | ADGEN_0171612 | Event1 | C:\Windows\System32\cmdkey.exe | cmd: cmdkey  /list | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 5 | ADGEN_0106402 | Event1 | C:\Windows\System32\rundll32.exe | cmd: C:\WINDOWS\system32\rundll32.exe /d acproxy.dll,PerformAutochkOperations | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 6 | ADGEN_0011557 | Event1 | C:\Windows\System32\net1.exe | cmd: C:\Windows\system32\net1  user <USERS_8> | ConnectPipe | C:\Windows\system32\net1.exe | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 7 | ADGEN_0041514 | Event3 | C:\Users\<USERS_5>\Downloads\a.exe | Event3 | C:\Users\<USERS_5>\Downloads\a.exe | Event3 | C:\Users\<USERS_5>\Downloads\a.exe | Event3 | C:\Users\<USE | MAL | T1059 | parent_chain | terminate_process |  |  |  |  |
| 8 | ADGEN_0035204 | Event1116 | Event1116 | Event1117 | Event1116 | Event1117 | BEN | none | parent_chain | no_action |  |  |  |  |
| 9 | ADGEN_0160715 | CreateKey | C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe | CreateKey | C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe | CreateKey | C:\W | MAL | T1059 | parent_chain | terminate_process |  |  |  |  |
| 10 | ADGEN_0170067 | Event1 | C:\Windows\System32\cmd.exe | cmd: "cmd.exe" /c "powershell -exec bypass -e SQBuAHYAbwBrAGUALQBXAG0AaQBNAGUAdABoAG8AZAAgAC0AUABhAHQAaAAgAHcAaQBuADMAMgB | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 11 | ADGEN_0045097 | Event1 | C:\Windows\System32\rundll32.exe | cmd: "C:\WINDOWS\system32\rundll32.exe" "C:\WINDOWS\SYSTEM32\EDGEHTML.dll",#141 Microsoft.Advertising.Xaml_8wekyb3d8 | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 12 | ADGEN_0012921 | ConnectPipe | C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe | ConnectPipe | C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe | Event22 | C: | MAL | T1059 | parent_chain | terminate_process |  |  |  |  |
| 13 | ADGEN_0030745 | Event100 | Event129 | Event200 | Event201 | Event102 | BEN | none | parent_chain | no_action |  |  |  |  |
| 14 | ADGEN_0108980 | Event1 | C:\Windows\System32\nslookup.exe | cmd: nslookup  <IP_LOCAL_843> | CreateKey | C:\Windows\system32\nslookup.exe | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 15 | ADGEN_0029882 | Event1 | C:\Windows\System32\rundll32.exe | cmd: C:\WINDOWS\system32\rundll32.exe /d acproxy.dll,PerformAutochkOperations | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 16 | ADGEN_0182424 | Event1 | C:\Windows\System32\conhost.exe | cmd: \??\C:\Windows\system32\conhost.exe 0xffffffff -ForceV1 | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 17 | ADGEN_0043971 | Event200 | Event201 | Event102 | Event301 | Event318 | BEN | none | parent_chain | no_action |  |  |  |  |
| 18 | ADGEN_0046569 | Event200 | Event201 | Event102 | Event301 | Event318 | BEN | none | parent_chain | no_action |  |  |  |  |
| 19 | ADGEN_0114347 | Event1 | C:\Windows\System32\conhost.exe | cmd: \??\C:\Windows\system32\conhost.exe 0xffffffff -ForceV1 | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 20 | ADGEN_0075721 | Event1 | C:\Windows\System32\conhost.exe | cmd: \??\C:\Windows\system32\conhost.exe 0xffffffff -ForceV1 | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 21 | ADGEN_0078816 | Event5860 | Event5860 | Event5860 | Event5858 | Event5858 | BEN | none | parent_chain | no_action |  |  |  |  |
| 22 | ADGEN_0078409 | Event1 | C:\Windows\System32\conhost.exe | cmd: \??\C:\Windows\system32\conhost.exe 0xffffffff -ForceV1 | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 23 | ADGEN_0048568 | CreatePipe | C:\Windows\system32\gpupdate.exe | Event10 | C:\Windows\system32\gpupdate.exe | Event8 | C:\Windows\System32\gpupdate.exe | MAL | T1059 | parent_chain | terminate_process |  |  |  |  |
| 24 | ADGEN_0048233 | Event1 | C:\Windows\System32\conhost.exe | cmd: \??\C:\Windows\system32\conhost.exe 0xffffffff -ForceV1 | Event5007 | Event5007 | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 25 | ADGEN_0059379 | ConnectPipe | C:\WINDOWS\system32\wbem\wmiprvse.exe | ConnectPipe | C:\WINDOWS\system32\wbem\wmiprvse.exe | ConnectPipe | C:\WINDOWS\system32\wbem\wmiprvse.exe  | MAL | T1059 | parent_chain | terminate_process |  |  |  |  |
| 26 | ADGEN_0030933 | Event129 | Event102 | Event201 | Event301 | Event318 | BEN | none | parent_chain | no_action |  |  |  |  |
| 27 | ADGEN_0020651 | Event1 | C:\Windows\System32\conhost.exe | cmd: \??\C:\Windows\system32\conhost.exe 0xffffffff -ForceV1 | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 28 | ADGEN_0041094 | ConnectPipe | C:\Windows\System32\svchost.exe | SetValue | C:\Windows\System32\svchost.exe | ConnectPipe | C:\Windows\System32\svchost.exe | CreateKey | C:\Wind | MAL | T1059 | parent_chain | terminate_process |  |  |  |  |
| 29 | ADGEN_0178537 | Event1 | C:\Windows\System32\schtasks.exe | cmd: schtasks  /create /tn "T1053_005_OnStartup" /sc onstart /ru system /tr "cmd.exe /c calc.exe" | Event7 | C:\Wind | MAL | T1053 | cmdline | terminate_process |  |  |  |  |
| 30 | ADGEN_0018623 | Event7 | \\DESKTOP-4PVPS6E\ADMIN$\841920a.exe | CreatePipe | \\DESKTOP-4PVPS6E\ADMIN$\841920a.exe | MAL | T1059 | parent_chain | terminate_process |  |  |  |  |
| 31 | ADGEN_0144178 | Event1 | C:\Windows\System32\conhost.exe | cmd: \??\C:\Windows\system32\conhost.exe 0xffffffff -ForceV1 | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 32 | ADGEN_0122876 | Event1 | C:\Windows\System32\schtasks.exe | cmd: schtasks.exe  /Create /F /TN "ATOMIC-T1053.005" /TR "cmd /c start /min \"\" powershell.exe -Command IEX([System | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 33 | ADGEN_0012418 | Event101 | Event1013 | Event101 | Event101 | Event1001 | BEN | none | parent_chain | no_action |  |  |  |  |
| 34 | ADGEN_0050754 | CreateKey | C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe | CreateKey | C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe | CreateKey | C:\W | MAL | T1059 | parent_chain | terminate_process |  |  |  |  |
| 35 | ADGEN_0022611 | Event1 | C:\Windows\System32\cmd.exe | cmd: C:\Windows\system32\cmd.exe /c C:\Windows\system32\powercfg.exe /getactivescheme > C:\Windows\TEMP\PowerPlan.log | E | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 36 | ADGEN_0054105 | Event7 | C:\Windows\System32\schtasks.exe | Event1 | C:\Windows\System32\schtasks.exe | cmd: schtasks  /create /ru system /sc daily /tr "cmd /c powershell.exe - | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 37 | ADGEN_0031386 | Event4672 | Event4672 | Event4672 | Event4672 | Event4672 | BEN | none | parent_chain | no_action |  |  |  |  |
| 38 | ADGEN_0031799 | Event1 | C:\Windows\System32\cmd.exe | cmd: C:\Windows\system32\cmd.exe /c echo a769d2bf136 > \\.\pipe\ea6a44 | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
| 39 | ADGEN_0023123 | Event4104 | Event4104 | Event4104 | Event4104 | Event4104 | BEN | none | parent_chain | no_action |  |  |  |  |
| 40 | ADGEN_0048727 | Event1 | C:\Windows\System32\rundll32.exe | cmd: C:\Windows\system32\rundll32.exe /d acproxy.dll,PerformAutochkOperations | MAL | T1059 | cmdline | terminate_process |  |  |  |  |
