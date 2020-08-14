//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
package edu.iu.dsc.tws.local.util;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.security.SecureClassLoader;
import java.util.HashSet;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.Twister2Job;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.exceptions.Twister2RuntimeException;

/**
 * This class loader will be used to virtually create isolated contexts for Worker instances.
 * This will,
 * <ul>
 * <li>Exclude all classes in java. or sun. packages</li>
 * <li>Exclude {@link Twister2Job}, {@link Config} and {@link Config.Builder},
 * since they should be passed from parent loader to this loader</li>
 * <li>Exclude edu.iu.dsc.tws.proto package, since it has lot of inner classes and not much
 * useful for the local runner</li>
 * </ul>
 */
public class LocalClassLoader extends SecureClassLoader {

  private static final Logger LOG = Logger.getLogger(LocalClassLoader.class.getName());

  private static final Set<String> APP_SPECIFIC_CLASS_EXCLUSIONS = new HashSet<>();
  private static final Set<String> APP_SPECIFIC_PACKAGE_EXCLUSIONS = new HashSet<>();

  private Set<String> twsClassesToExclude = new HashSet<>();
  private Set<String> twsPackagesToExclude = new HashSet<>();
  private Set<String> classesToLoad = new HashSet<>();

  public LocalClassLoader(ClassLoader parent) {
    super(parent);
    // delegating following classes to parent class loader
    twsClassesToExclude.add(Twister2Job.class.getName());
    twsClassesToExclude.add(Config.class.getName());
    twsClassesToExclude.add(Config.Builder.class.getName());
    //twsClassesToExclude.add(TBaseGraph.class.getName()); This shouldn't be uncommented

    twsClassesToExclude.addAll(APP_SPECIFIC_CLASS_EXCLUSIONS);

    // delegating following packages to parent class loader
    twsPackagesToExclude.add("edu.iu.dsc.tws.proto");
    // twsPackagesToExclude.add("edu.iu.dsc.tws"); This shouldn't be uncommented
    twsPackagesToExclude.add("jep"); // to support python debugging
    twsPackagesToExclude.add("edu.iu.dsc.tws.python.processors.JepInstance");

    twsPackagesToExclude.addAll(APP_SPECIFIC_PACKAGE_EXCLUSIONS);
  }

  public static <T> void excludeClass(Class<T> clazz) {
    APP_SPECIFIC_CLASS_EXCLUSIONS.add(clazz.getName());
  }

  public static void excludeClass(String fullyQualifiedClassName) {
    APP_SPECIFIC_CLASS_EXCLUSIONS.add(fullyQualifiedClassName);
  }

  public static void excludePackage(String packageName) {
    APP_SPECIFIC_PACKAGE_EXCLUSIONS.add(packageName);
  }

  public void addJobClass(String jobClass) {
    this.classesToLoad.add(jobClass);
  }

  public boolean excludedPackage(String className) {
    for (String s : this.twsPackagesToExclude) {
      if (className.contains(s)) {
        return true;
      }
    }
    return false;
  }

  @Override
  public Class<?> loadClass(String name) throws ClassNotFoundException {
    if (!name.startsWith("java.")
        && !name.startsWith("sun.")
        && !twsClassesToExclude.contains(name)
        && !this.excludedPackage(name)) {
      InputStream is = getResourceAsStream(name.replace(".", "/") + ".class");
      try {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        int readBytes;
        while ((readBytes = is.read(buffer)) != -1) {
          baos.write(buffer, 0, readBytes);
        }
        byte[] bytes = baos.toByteArray();
        return defineClass(name, bytes, 0, bytes.length);
      } catch (NullPointerException nex) {
        throw new ClassNotFoundException(name);
      } catch (Exception e) {
        LOG.log(Level.SEVERE, "Error loading " + name, e);
        throw new Twister2RuntimeException(e);
      }
    } else {
      return super.loadClass(name);
    }
  }
}
