### AQL files

The **aql\*.json.in** files are templates for retrieval of packages from
the lkeb-artifactory. They are completed by the **CMake** **configure_file** function.

The following variants are available

| File name | Purpose | Variables |
--- | --- | ---
aql_multi.json.in | Retrieve a multi-build i.e. Debug + Release in one package | <ul><li>package_builder: usually lkeb</li><li>package_name: e.g. flann</li><li>package_version: e.g. 1.8.5</li><li>os_name: Windows,Macos or Linux</li><li>compiler_name: apple-clang, Visual Studio, gcc</li><li>compiler_version</li></ul>
aql_single.json.in | Retrieve a single build i.e. Debug or Release in one package (these have come from conan_center) | <ul><li>package_name: e.g. flann</li><li>package_version: e.g. 1.8.5</li><li>os_name: Windows,Macos or Linux</li><li>compiler_name: apple-clang, Visual Studio, gcc</li><li>compiler_version</li><li>build_runtime_type: either <ol><li>Windows: "conan.settings.compiler.runtime":{"\$eq" : "_MD or MDd_"}</li><li>Linux/Macos: "conan.settings.build_type":{"\$eq" : "_Debug or Release_"}</li></ol></li><li>shared: True or False</li></ul>

