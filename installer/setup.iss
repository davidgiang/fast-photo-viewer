#define MyAppName "Fast Photo Viewer"
#define MyAppPublisher "David Giang"
#define MyAppURL "https://github.com/davidgiang1/fast-photo-viewer"
#define MyAppExeName "fast-photo-viewer.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\FastPhotoViewer
DefaultGroupName={#MyAppName}
OutputDir=..\output
OutputBaseFilename=FastPhotoViewer-{#MyAppVersion}-setup
Compression=lzma2
SolidCompression=yes
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "associateimages"; Description: "Associate image files (jpg, png, bmp, webp, gif, tiff, svg)"; GroupDescription: "File associations:"
Name: "associatevideos"; Description: "Associate video files (mp4, mkv, avi, mov, wmv, etc.)"; GroupDescription: "File associations:"; Flags: unchecked

[Files]
Source: "..\target\release\fast-photo-viewer.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dlls\avcodec-61.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dlls\avdevice-61.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dlls\avfilter-10.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dlls\avformat-61.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dlls\avutil-59.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dlls\postproc-58.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dlls\swresample-5.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dlls\swscale-8.dll"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; ProgID for images
Root: HKLM; Subkey: "SOFTWARE\Classes\FastPhotoViewer.Image"; ValueType: string; ValueData: "Fast Photo Viewer Image"; Flags: uninsdeletekey
Root: HKLM; Subkey: "SOFTWARE\Classes\FastPhotoViewer.Image\DefaultIcon"; ValueType: string; ValueData: "{app}\{#MyAppExeName},0"
Root: HKLM; Subkey: "SOFTWARE\Classes\FastPhotoViewer.Image\shell\open\command"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

; ProgID for videos
Root: HKLM; Subkey: "SOFTWARE\Classes\FastPhotoViewer.Video"; ValueType: string; ValueData: "Fast Photo Viewer Video"; Flags: uninsdeletekey
Root: HKLM; Subkey: "SOFTWARE\Classes\FastPhotoViewer.Video\DefaultIcon"; ValueType: string; ValueData: "{app}\{#MyAppExeName},0"
Root: HKLM; Subkey: "SOFTWARE\Classes\FastPhotoViewer.Video\shell\open\command"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

; Image file associations
Root: HKLM; Subkey: "SOFTWARE\Classes\.jpg\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Image"; Flags: uninsdeletevalue; Tasks: associateimages
Root: HKLM; Subkey: "SOFTWARE\Classes\.jpeg\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Image"; Flags: uninsdeletevalue; Tasks: associateimages
Root: HKLM; Subkey: "SOFTWARE\Classes\.png\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Image"; Flags: uninsdeletevalue; Tasks: associateimages
Root: HKLM; Subkey: "SOFTWARE\Classes\.bmp\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Image"; Flags: uninsdeletevalue; Tasks: associateimages
Root: HKLM; Subkey: "SOFTWARE\Classes\.webp\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Image"; Flags: uninsdeletevalue; Tasks: associateimages
Root: HKLM; Subkey: "SOFTWARE\Classes\.gif\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Image"; Flags: uninsdeletevalue; Tasks: associateimages
Root: HKLM; Subkey: "SOFTWARE\Classes\.tiff\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Image"; Flags: uninsdeletevalue; Tasks: associateimages
Root: HKLM; Subkey: "SOFTWARE\Classes\.ico\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Image"; Flags: uninsdeletevalue; Tasks: associateimages
Root: HKLM; Subkey: "SOFTWARE\Classes\.svg\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Image"; Flags: uninsdeletevalue; Tasks: associateimages

; Video file associations
Root: HKLM; Subkey: "SOFTWARE\Classes\.mp4\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Video"; Flags: uninsdeletevalue; Tasks: associatevideos
Root: HKLM; Subkey: "SOFTWARE\Classes\.mkv\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Video"; Flags: uninsdeletevalue; Tasks: associatevideos
Root: HKLM; Subkey: "SOFTWARE\Classes\.avi\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Video"; Flags: uninsdeletevalue; Tasks: associatevideos
Root: HKLM; Subkey: "SOFTWARE\Classes\.mov\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Video"; Flags: uninsdeletevalue; Tasks: associatevideos
Root: HKLM; Subkey: "SOFTWARE\Classes\.wmv\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Video"; Flags: uninsdeletevalue; Tasks: associatevideos
Root: HKLM; Subkey: "SOFTWARE\Classes\.flv\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Video"; Flags: uninsdeletevalue; Tasks: associatevideos
Root: HKLM; Subkey: "SOFTWARE\Classes\.webm\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Video"; Flags: uninsdeletevalue; Tasks: associatevideos
Root: HKLM; Subkey: "SOFTWARE\Classes\.m4v\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Video"; Flags: uninsdeletevalue; Tasks: associatevideos
Root: HKLM; Subkey: "SOFTWARE\Classes\.mpg\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Video"; Flags: uninsdeletevalue; Tasks: associatevideos
Root: HKLM; Subkey: "SOFTWARE\Classes\.mpeg\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Video"; Flags: uninsdeletevalue; Tasks: associatevideos
Root: HKLM; Subkey: "SOFTWARE\Classes\.3gp\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Video"; Flags: uninsdeletevalue; Tasks: associatevideos
Root: HKLM; Subkey: "SOFTWARE\Classes\.ogv\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Video"; Flags: uninsdeletevalue; Tasks: associatevideos
Root: HKLM; Subkey: "SOFTWARE\Classes\.vob\OpenWithProgids"; ValueType: string; ValueName: "FastPhotoViewer.Video"; Flags: uninsdeletevalue; Tasks: associatevideos

; Register in Windows Default Programs / Capabilities
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities"; ValueType: string; ValueName: "ApplicationName"; ValueData: "{#MyAppName}"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities"; ValueType: string; ValueName: "ApplicationDescription"; ValueData: "Fast lightweight image and video viewer"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities\FileAssociations"; ValueType: string; ValueName: ".jpg"; ValueData: "FastPhotoViewer.Image"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities\FileAssociations"; ValueType: string; ValueName: ".jpeg"; ValueData: "FastPhotoViewer.Image"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities\FileAssociations"; ValueType: string; ValueName: ".png"; ValueData: "FastPhotoViewer.Image"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities\FileAssociations"; ValueType: string; ValueName: ".bmp"; ValueData: "FastPhotoViewer.Image"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities\FileAssociations"; ValueType: string; ValueName: ".webp"; ValueData: "FastPhotoViewer.Image"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities\FileAssociations"; ValueType: string; ValueName: ".gif"; ValueData: "FastPhotoViewer.Image"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities\FileAssociations"; ValueType: string; ValueName: ".tiff"; ValueData: "FastPhotoViewer.Image"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities\FileAssociations"; ValueType: string; ValueName: ".ico"; ValueData: "FastPhotoViewer.Image"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities\FileAssociations"; ValueType: string; ValueName: ".svg"; ValueData: "FastPhotoViewer.Image"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities\FileAssociations"; ValueType: string; ValueName: ".mp4"; ValueData: "FastPhotoViewer.Video"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities\FileAssociations"; ValueType: string; ValueName: ".mkv"; ValueData: "FastPhotoViewer.Video"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities\FileAssociations"; ValueType: string; ValueName: ".avi"; ValueData: "FastPhotoViewer.Video"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities\FileAssociations"; ValueType: string; ValueName: ".mov"; ValueData: "FastPhotoViewer.Video"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities\FileAssociations"; ValueType: string; ValueName: ".wmv"; ValueData: "FastPhotoViewer.Video"
Root: HKLM; Subkey: "SOFTWARE\FastPhotoViewer\Capabilities\FileAssociations"; ValueType: string; ValueName: ".webm"; ValueData: "FastPhotoViewer.Video"
Root: HKLM; Subkey: "SOFTWARE\RegisteredApplications"; ValueType: string; ValueName: "FastPhotoViewer"; ValueData: "SOFTWARE\FastPhotoViewer\Capabilities"; Flags: uninsdeletevalue

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
