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
package edu.iu.dsc.tws.dl.module;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.data.Table;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

@SuppressWarnings("NeedBraces")
public abstract class Container<A extends Activity> extends AbstractModule<A> {

  /**
   * Module status. It is useful for modules like dropout/batch normalization
   */
  protected boolean train = true;


  protected String line = "\n";
  public List<AbstractModule> modules = new ArrayList<AbstractModule>();

  @Override
  public List<AbstractModule> getModules() {
    return modules;
  }

  @Override
  public void setModules(List<AbstractModule> modules) {
    this.modules = modules;
  }

  @Override
  public final AbstractModule training() {
    train = true;
    modules.forEach(abstractModule -> abstractModule.training());
    return this;
  }

  @Override
  public void reset() {
    modules.forEach(abstractModule -> abstractModule.reset());
  }

//  final override def evaluate(): this.type = {
//    train = false
//    modules.foreach(_.evaluate())
//    this
//  }
//
//  final override def checkEngineType(): this.type = {
//    modules.foreach(_.checkEngineType())
//    this
//  }


  @Override
  public Object[] getTimes() {
//    Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
//      if (modules.isEmpty) {
//        return Array((this, forwardTime, backwardTime))
//      }
//      val subModuleTimes = this.modules.flatMap(_.getTimes()).toArray
//
//      val (subModuleForward, subModuleBackward) = Utils.calculateFwdBwdTime(subModuleTimes)
//
//      subModuleTimes ++ Array((this, this.forwardTime - subModuleForward,
//          this.backwardTime - subModuleBackward))
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public void resetTimes() {
    super.resetTimes();
    modules.forEach(abstractModule -> abstractModule.resetTimes());
  }

  @Override
  public TensorArrayPair parameters() {
    List<Tensor> weights = new ArrayList<>();
    List<Tensor> gradWeights = new ArrayList<>();
    modules.forEach(m -> {
      TensorArrayPair params = m.parameters();
      if (params != null) {
        for (Tensor tensor : params.getValue0()) {
          weights.add(tensor);
        }
        for (Tensor tensor : params.getValue1()) {
          gradWeights.add(tensor);
        }
      }
    });
    return new TensorArrayPair(weights.toArray(new Tensor[0]), gradWeights.toArray(new Tensor[0]));
  }

  @Override
  public Tensor[] getExtraParameter() {
    return new Tensor[0];
  }

  @Override
  public Table getParametersTable() {
    return super.getParametersTable();
  }

  public List<AbstractModule> findModules(String moduleType) {
    List<AbstractModule> nodes = new ArrayList<>();
    if (getName(this) == moduleType) {
      nodes.add(this);
    }
    for (AbstractModule module : modules) {
      if (module instanceof Container) {
        nodes.addAll(((Container) module).findModules(moduleType));
      } else {
        if (getName(module) == moduleType) {
          nodes.add(module);
        }
      }
    }
    return nodes;
  }

  private String getName(AbstractModule module) {
    String[] x = module.getClass().getName().split("\\.");
    return x[x.length - 1];
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    if (!super.equals(o)) return false;
    Container container = (Container) o;
    return modules.equals(container.modules);
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), modules);
  }

  @Override
  public AbstractModule setScaleW(double w) {
    modules.forEach(module -> module.setScaleW(w));
    return this;
  }

  @Override
  public AbstractModule setScaleB(double b) {
    modules.forEach(module -> module.setScaleB(b));
    return this;
  }

  @Override
  public AbstractModule freeze(String[] names) {
    if (names == null || names.length == 0) {
      modules.forEach(module -> module.freeze(null));
    } else {
      boolean found = false;
      for (String name : names) {
        if (apply(name) != null) {
          apply(name).freeze(null);
          found = true;
        }
      }
      if (!found) {
        throw new IllegalStateException("cannot match module named $name");
      }
    }
    return this;
  }

  @Override
  public AbstractModule unFreeze(String[] names) {
    if (names == null || names.length == 0) {
      modules.forEach(module -> module.unFreeze(null));
    } else {
      boolean found = false;
      for (String name : names) {
        if (apply(name) != null) {
          apply(name).unFreeze(null);
          found = true;
        }
      }
      if (!found) {
        throw new IllegalStateException("cannot match module named $name");
      }
    }
    return this;
  }

  @Override
  public AbstractModule apply(String name) {
    if (this.getName() == name) {
      return this;
    } else {
      boolean found = false;
      AbstractModule find = null;
      for (AbstractModule module : modules) {
        if (module.apply(name) != null) {
          if (!found) {
            found = true;
            find = module.apply(name);
          } else {
            throw new IllegalStateException("find multiple modules with same name");
          }
        }
      }

      return find;
    }
  }

//  /**
//   * Check if some module is duplicated in the model
//   */
//  private[bigdl] override final def checkDuplicate(
//      record: mutable.HashSet[Int] = mutable.HashSet()
//  ): Unit = {
//    super.checkDuplicate(record)
//    if (!skipDuplicateCheck()) modules.foreach(_.checkDuplicate(record))
//  }
//
//  override def release(): Unit = {
//    modules.foreach(_.release())
//  }
//
  @Override
  public void asyncGradient() {
  }

  @Override
  protected void updateParameter() {
  }
}
