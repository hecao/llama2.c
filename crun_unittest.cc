# include <gtest/gtest.h>
# include "crun.h"

TEST(MyClassTest, Add) {
    MyClass my_class;
    EXPECT_EQ(my_class.Add(1, 2), 3);
}