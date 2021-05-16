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
package edu.iu.dsc.tws.dl.utils;

import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Parameter;
import java.util.HashMap;
import java.util.Map;

import edu.iu.dsc.tws.dl.module.AbstractModule;
import edu.iu.dsc.tws.dl.utils.graph.IRElement;
import edu.iu.dsc.tws.dl.utils.graph.IROperator;
import edu.iu.dsc.tws.dl.utils.pair.TensorPair;

public final class ReflectionUtils {

  private ReflectionUtils() {
  }

  public static Class findClass(String name) {
    try {
      return Class.forName(name);
    } catch (ClassNotFoundException e) {
      return null;
    } catch (Exception ex) {
      throw ex;
    }
  }

  public static IRElement reflectToIR(AbstractModule source, Class target) {
    Map<String, Object> nameAndValues = getFieldNameAndValues(source);
    IROperator op = (IROperator) ReflectionUtils.reflection(target, nameAndValues);
    TensorPair weightsAndBias;
    if (source.parameters() != null) {
      weightsAndBias = source.getParameters();
    } else {
      weightsAndBias = new TensorPair(null, null);
    }
    IRElement element = new IRElement(source.getName(), op, weightsAndBias.getValue0(),
        weightsAndBias.getValue1());
    return element;
  }

  /**
   * Create a Module object of target Class by mirroring the given IR element
   *
   * @param source the source IR element
   * @param target the target class we want to make an object from
   * @return the target instance of type Module
   * @tparam T
   */
  public static AbstractModule reflectFromIR(IRElement source, Class target) {
    Map<String, Object> nameAndValues = getFieldNameAndValues(source.getOp());

    AbstractModule blasLayer = (AbstractModule) reflection(target, nameAndValues);

    if (blasLayer.parameters() != null) {
      TensorPair params = blasLayer.getParameters();
      TensorPair params2 = source.getParameters();
      if (params2.getValue0() != null) {
        params.getValue0().copy(params2.getValue0());
        source.setWeights(params.getValue0());
      }
      if (params2.getValue1() != null) {
        params.getValue1().copy(params2.getValue1());
        source.setGradWeights(params.getValue1());
      }
    }

    if (!source.getName().equals("")) {
      blasLayer.setName(source.getName());
    }
//    if (blasLayer.isInstanceOf[MklInt8Convertible]) {
//      setScales(source, blasLayer.asInstanceOf[MklInt8Convertible])
//    }

    return blasLayer;
  }

  /**
   * Get key value map from input object,
   * field name of the object as key, its reference as value
   *
   * @param o input object
   * @return A map which field name as key and field refernece as value
   */
  private static Map<String, Object> getFieldNameAndValues(Object o) {
    Class<?> c = o.getClass();
    Field[] fields = c.getDeclaredFields();
    Field[] superFields = c.getSuperclass().getDeclaredFields();
    Map<String, Object> result = new HashMap<>();
    try {
      for (Field field : fields) {
        field.setAccessible(true);
        result.put(field.getName(), field.get(o));
      }

      for (Field field : superFields) {
        field.setAccessible(true);
        result.put(field.getName(), field.get(o));
      }

    } catch (IllegalAccessException e) {
      e.printStackTrace();
    }
    return result;
  }

  public static Object reflection(Class target, Map<String, Object> nameAndValues) {

    try {
      Constructor[] ctor = target.getConstructors();
      Class[] paramList = ctor[0].getParameterTypes();
      Parameter[] parameters = ctor[0].getParameters();
      Object[] params = new Object[paramList.length];
      for (int i = 0; i < params.length; i++) {
        String name = parameters[i].getName();
        if (nameAndValues.containsKey(name)) {
          params[i] = nameAndValues.get(name);
        } else if (parameters[i].getType().isPrimitive()) {
          params[i] = getDefaultValue(parameters[i].getType());
        } else {
          params[i] = null;
        }

      }
      return ctor[0].newInstance(params);
    } catch (Exception ex) {
      throw new RuntimeException(ex);
    }
  }

  private static <T> T getDefaultValue(Class<T> clazz) {
    return (T) Array.get(Array.newInstance(clazz, 1), 0);
  }
}
